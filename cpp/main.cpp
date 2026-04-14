/*
 * main.cpp  —  MITC6 Shell FEM driver (input-file driven)
 *
 * Usage:  ./mitcs6 [input_file]   (default: plate.inp)
 */
#include "mitcs6.h"
#include "input_parser.h"
#include <gmsh.h>
#include <cstdio>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[])
{
    std::string inp = (argc >= 2) ? argv[1] : "plate.inp";
    gmsh::initialize();
    try {
        // ── parse ─────────────────────────────────────────────────────────────
        std::printf("Reading: %s\n", inp.c_str());
        ProblemDef pd  = parse_input(inp);
        print_problem_def(pd);
        Config     cfg = to_config(pd);

        // ── constitutive matrix ───────────────────────────────────────────────
        ABD abd = cfg.plies.empty()
                  ? isotropic_ABD(cfg.E, cfg.nu, cfg.thickness)
                  : compute_ABD(cfg.plies);
        double t_eff = cfg.total_thickness();

        // ── geometry ──────────────────────────────────────────────────────────
        std::printf("\n=== Geometry (%zu cutout(s)) ===\n", cfg.cutouts.size());
        create_geometry(cfg);
        if (cfg.verify_geometry_only) {
            std::printf("Geometry OK — stopping.\n");
            gmsh::finalize(); return 0;
        }

        // ── mesh ──────────────────────────────────────────────────────────────
        std::printf("=== Mesh ===\n");
        Mesh mesh = create_mesh(cfg);
        build_adjacency(mesh);
        std::printf("  Nodes %d  Elements %d  DOFs %d  (6 DOF/node)\n",
                    mesh.n_nodes(), mesh.n_elems(), mesh.n_dof());

        // ── stiffness ─────────────────────────────────────────────────────────
        std::printf("=== Stiffness ===\n");
        SpMat K = assemble_stiffness(mesh, abd);
        // Numerical regularisation: ε·I to remove residual zero-energy modes
        for (int i = 0; i < mesh.n_dof(); ++i)
            K.coeffRef(i, i) += 1e-12 * abd.A.norm();

        // ── loads ─────────────────────────────────────────────────────────────
        std::printf("=== Loads ===\n");
        Eigen::VectorXd F = Eigen::VectorXd::Zero(mesh.n_dof());
        if (pd.pressure != 0.0) {
            F += assemble_pressure_load(mesh, pd.pressure);
            std::printf("  Pressure %.4f MPa\n", pd.pressure);
        }
        apply_problem_loads(pd, mesh, F);
        std::printf("  |F| = %.4e N\n", F.norm());

        // ── boundary conditions ───────────────────────────────────────────────
        std::printf("=== BCs ===\n");
        std::vector<int> fixed;
        apply_problem_bcs(pd, mesh, fixed);

        // Suppress rigid-body in-plane translations if no in-plane BC present
        bool has_u=false, has_v=false;
        for (auto& bc : pd.bcs) {
            int m = bc_mask_public(bc.type);
            if (m & DOF_U) has_u = true;
            if (m & DOF_V) has_v = true;
        }
        auto pin_one = [&](const std::string& grp, int dof) {
            try {
                auto nodes = nodes_on_group(1, grp);
                if (!nodes.empty())
                    fixed.push_back(6 * mesh.node_id.at(nodes[0]) + dof);
            } catch (...) {}
        };
        if (!has_u) pin_one("RIGHT",  0);
        if (!has_v) pin_one("BOTTOM", 1);

        apply_bcs(K, F, fixed);
        std::printf("  Fixed %zu  Free %d\n",
                    fixed.size(), mesh.n_dof() - (int)fixed.size());

        // ── solve ─────────────────────────────────────────────────────────────
        std::printf("=== Solve ===\n");
        Eigen::VectorXd U = solve_system(K, F);
        if (!U.allFinite())
            throw std::runtime_error("Solution contains NaN/Inf.");

        // ── stress recovery ───────────────────────────────────────────────────
        std::printf("=== Stress recovery (centroid + SPR) ===\n");
        auto resp  = recover_all(mesh, U, abd);
        auto nodal = spr_recovery(mesh, resp);   // SPR nodal stresses
        print_summary(mesh, U, resp);

        // ── export ────────────────────────────────────────────────────────────
        std::printf("=== Export ===\n");
        export_vtk(cfg.out_base + ".vtk", mesh, U, resp, nodal, t_eff);
        std::printf("Done → %s.vtk\n", cfg.out_base.c_str());
        std::printf("  VTK fields: spr_Nx/Ny/Nxy/Mx/My/Mxy/Qx/Qy (SPR nodal)\n");
        std::printf("              spr_top/bot_sigma1/sigma2/vonMises (SPR surface)\n");
        std::printf("              centroid_Nx/Ny/Mx/vm_top/vm_bot (element avg)\n");
    }
    catch (const std::exception& ex) {
        std::fprintf(stderr, "\nERROR: %s\n", ex.what());
        gmsh::finalize(); return 1;
    }
    gmsh::finalize();
    return 0;
}
