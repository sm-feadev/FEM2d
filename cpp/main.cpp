/*
 * main.cpp  —  MITC6 Shell FEM driver
 *
 * Demonstrates:
 *   • single cutout   (rounded rectangle)
 *   • two cutouts     (circle + ellipse)
 *   • three cutouts   (rectangle + circle + rounded_rectangle)
 *
 * Select the scenario by setting SCENARIO = 1 / 2 / 3 below.
 * All results are written to plate_results.vtk (open in ParaView).
 */

#include "mitcs6.h"
#include <gmsh.h>
#include <cstdio>
#include <stdexcept>

// ── choose demo scenario ──────────────────────────────────────────────────────
static constexpr int SCENARIO = 1;

static Config make_config()
{
    Config cfg;
    cfg.plate_w   = 300.0;
    cfg.plate_h   = 200.0;
    cfg.mesh_size =  10.0;
    cfg.E         = 210000.0;   // steel [MPa]
    cfg.nu        =      0.30;
    cfg.thickness =     10.0;   // [mm]
    cfg.pressure  =     -0.1;   // lateral pressure, –ve = downward [MPa]
    cfg.msh_file  = "plate_mitc6.msh";
    cfg.out_base  = "plate_results";

    if constexpr (SCENARIO == 1) {
        // ── single rounded-rectangle cutout ──────────────────────────────────
        cfg.cutouts.push_back({CutoutType::RoundedRectangle,
                               150.0, 100.0,   // cx, cy
                               120.0,  60.0,   // w,  h
                                10.0});         // r
    }
    else if constexpr (SCENARIO == 2) {
        // ── two cutouts: circle (left) + ellipse (right) ─────────────────────
        cfg.cutouts.push_back({CutoutType::Circle,
                               100.0, 100.0,   // cx, cy
                                 0.0,   0.0,   // w, h unused for circle
                                30.0});         // r = radius
        cfg.cutouts.push_back({CutoutType::Ellipse,
                               210.0, 100.0,
                                60.0,  30.0,   // w=2a, h=2b
                                 0.0});
    }
    else if constexpr (SCENARIO == 3) {
        // ── three cutouts ────────────────────────────────────────────────────
        cfg.cutouts.push_back({CutoutType::Rectangle,
                                80.0, 100.0,
                                40.0,  40.0,
                                 0.0});
        cfg.cutouts.push_back({CutoutType::Circle,
                               165.0, 100.0,
                                 0.0,   0.0,
                                25.0});
        cfg.cutouts.push_back({CutoutType::RoundedRectangle,
                               240.0, 100.0,
                                60.0,  50.0,
                                 8.0});
    }

    return cfg;
}

int main()
{
    gmsh::initialize();
    try {
        Config cfg = make_config();

        // ── geometry ─────────────────────────────────────────────────────────
        std::printf("=== Creating geometry (scenario %d, %zu cutout(s)) ===\n",
                    SCENARIO, cfg.cutouts.size());
        create_geometry(cfg);

        if (cfg.verify_geometry_only) {
            std::printf("Geometry OK — stopping (verify_geometry_only=true).\n");
            gmsh::finalize();
            return 0;
        }

        // ── mesh ─────────────────────────────────────────────────────────────
        std::printf("=== Meshing ===\n");
        Mesh mesh = create_mesh(cfg);
        std::printf("  Nodes   : %d\n", mesh.n_nodes());
        std::printf("  Elements: %d\n", mesh.n_elems());
        std::printf("  DOFs    : %d\n", mesh.n_dof());

        // ── stiffness ────────────────────────────────────────────────────────
        std::printf("=== Assembling stiffness ===\n");
        SpMat K = assemble_stiffness(mesh, cfg.E, cfg.nu, cfg.thickness);

        // Small numerical regularisation to remove zero-energy modes
        // K += ε·I   (ε ≪ smallest non-zero eigenvalue)
        for (int i = 0; i < mesh.n_dof(); ++i)
            K.coeffRef(i, i) += 1e-12 * cfg.E;

        // ── load: uniform lateral pressure ───────────────────────────────────
        std::printf("=== Assembling load vector ===\n");
        Eigen::VectorXd F = assemble_pressure_load(mesh, cfg.pressure);
        std::printf("  |F| = %.4e N\n", F.norm());

        // ── boundary conditions ───────────────────────────────────────────────
        // Left edge: fully clamped (all 5 DOFs)
        std::printf("=== Applying boundary conditions ===\n");
        std::vector<int> fixed;
        auto left_nodes = nodes_on_group(1, "LEFT");
        add_fixed_dofs(mesh, left_nodes, DOF_ALL, fixed);

        // Suppress in-plane rigid-body translations on remaining free edges:
        //   one u-DOF on RIGHT,  one v-DOF on BOTTOM
        auto right_nodes  = nodes_on_group(1, "RIGHT");
        auto bottom_nodes = nodes_on_group(1, "BOTTOM");
        if (!right_nodes.empty()) {
            int n = mesh.node_id.at(right_nodes[0]);
            fixed.push_back(5*n + 0);   // u
        }
        if (!bottom_nodes.empty()) {
            int n = mesh.node_id.at(bottom_nodes[0]);
            fixed.push_back(5*n + 1);   // v
        }

        apply_bcs(K, F, fixed);
        std::printf("  Fixed DOFs : %zu\n", fixed.size());
        std::printf("  Free DOFs  : %d\n",  mesh.n_dof() - static_cast<int>(fixed.size()));

        // ── solve ─────────────────────────────────────────────────────────────
        std::printf("=== Solving ===\n");
        Eigen::VectorXd U = solve_system(K, F);

        // Sanity check
        bool ok = U.allFinite();
        if (!ok) throw std::runtime_error("Solution contains NaN/Inf.");

        // ── stress recovery ───────────────────────────────────────────────────
        std::printf("=== Recovering element responses ===\n");
        auto resp = recover_all(mesh, U, cfg.E, cfg.nu, cfg.thickness);

        print_summary(mesh, U, resp);

        // ── export ─────────────────────────────────────────────────────────────
        std::printf("=== Exporting ===\n");
        export_vtk(cfg.out_base + ".vtk", mesh, U, resp, cfg.thickness);

        std::printf("Done.\n");
    }
    catch (const std::exception& ex) {
        std::fprintf(stderr, "ERROR: %s\n", ex.what());
        gmsh::finalize();
        return 1;
    }
    gmsh::finalize();
    return 0;
}
