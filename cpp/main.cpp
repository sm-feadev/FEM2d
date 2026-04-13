/*
 * main.cpp  —  MITC6 Shell FEM driver (input-file driven)
 *
 * Usage:
 *   ./mitcs6 plate.inp
 *
 * If no argument is given, looks for "plate.inp" in the current directory.
 */

#include "mitcs6.h"
#include "input_parser.h"
#include <gmsh.h>
#include <cstdio>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[])
{
    std::string inp_file = (argc >= 2) ? argv[1] : "plate.inp";

    gmsh::initialize();
    try {
        // ── parse input ───────────────────────────────────────────────────────
        std::printf("Reading: %s\n", inp_file.c_str());
        ProblemDef pd  = parse_input(inp_file);
        print_problem_def(pd);
        Config     cfg = to_config(pd);

        // ── geometry ──────────────────────────────────────────────────────────
        std::printf("\n=== Geometry (%zu cutout(s)) ===\n", cfg.cutouts.size());
        create_geometry(cfg);
        if (cfg.verify_geometry_only) {
            std::printf("Geometry OK — stopping.\n");
            gmsh::finalize();
            return 0;
        }

        // ── mesh ──────────────────────────────────────────────────────────────
        std::printf("=== Mesh ===\n");
        Mesh mesh = create_mesh(cfg);
        std::printf("  Nodes %d  Elements %d  DOFs %d\n",
                    mesh.n_nodes(), mesh.n_elems(), mesh.n_dof());

        // ── stiffness ─────────────────────────────────────────────────────────
        std::printf("=== Stiffness ===\n");
        SpMat K = assemble_stiffness(mesh, cfg.E, cfg.nu, cfg.thickness);
        for (int i = 0; i < mesh.n_dof(); ++i)
            K.coeffRef(i, i) += 1e-12 * cfg.E;

        // ── loads ─────────────────────────────────────────────────────────────
        std::printf("=== Loads ===\n");
        Eigen::VectorXd F = Eigen::VectorXd::Zero(mesh.n_dof());
        if (pd.pressure != 0.0) {
            Eigen::VectorXd Fp = assemble_pressure_load(mesh, pd.pressure);
            F += Fp;
            std::printf("  Pressure %.4f MPa  |Fp|=%.4e\n", pd.pressure, Fp.norm());
        }
        apply_problem_loads(pd, mesh, F);
        std::printf("  Total |F| = %.4e N\n", F.norm());

        // ── boundary conditions ───────────────────────────────────────────────
        std::printf("=== BCs ===\n");
        std::vector<int> fixed;
        apply_problem_bcs(pd, mesh, fixed);

        // Suppress any remaining in-plane rigid-body modes by pinning one node
        // on RIGHT (u) and BOTTOM (v) — only if no BC already fixes those DOFs.
        bool has_u_bc = false, has_v_bc = false;
        for (const auto& bc : pd.bcs) {
            int m = bc_mask_public(bc.type);
            if (m & DOF_U) has_u_bc = true;
            if (m & DOF_V) has_v_bc = true;
        }
        auto pin_one = [&](const std::string& grp, int dof){
            try {
                auto nodes = nodes_on_group(1, grp);
                if (!nodes.empty())
                    fixed.push_back(5 * mesh.node_id.at(nodes[0]) + dof);
            } catch (...) {}
        };
        if (!has_u_bc) pin_one("RIGHT",  0);
        if (!has_v_bc) pin_one("BOTTOM", 1);

        apply_bcs(K, F, fixed);
        std::printf("  Fixed %zu  Free %d\n",
                    fixed.size(), mesh.n_dof() - (int)fixed.size());

        // ── solve ─────────────────────────────────────────────────────────────
        std::printf("=== Solve ===\n");
        Eigen::VectorXd U = solve_system(K, F);
        if (!U.allFinite())
            throw std::runtime_error("Solution contains NaN/Inf.");

        // ── stress recovery ───────────────────────────────────────────────────
        std::printf("=== Stress recovery ===\n");
        auto resp = recover_all(mesh, U, cfg.E, cfg.nu, cfg.thickness);
        print_summary(mesh, U, resp);

        // ── export ────────────────────────────────────────────────────────────
        std::printf("=== Export ===\n");
        export_vtk(cfg.out_base + ".vtk", mesh, U, resp, cfg.thickness);
        std::printf("Done → %s.vtk\n", cfg.out_base.c_str());
    }
    catch (const std::exception& ex) {
        std::fprintf(stderr, "\nERROR: %s\n", ex.what());
        gmsh::finalize();
        return 1;
    }
    gmsh::finalize();
    return 0;
}
