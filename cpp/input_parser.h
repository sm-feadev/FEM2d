#pragma once
/*
 * input_parser.h  —  INI-style input file reader for MITC6 FEM solver
 *
 * Repeatable sections: [cutout]  [ply]  [bc]  [point_load]  [edge_load]
 *
 * Material selection
 * ──────────────────
 *   Isotropic (default):
 *     [material]
 *     E = 210000 | nu = 0.30 | thickness = 10.0
 *
 *   Laminate (overrides [material] when one or more [ply] sections present):
 *     [ply]
 *     E1=135000 | E2=10000 | G12=5000 | nu12=0.30 | angle=0 | tk=1.25
 *     (repeat for each ply; plies are stacked in order of appearance)
 *
 *   Built-in material presets (set in [material] preset = <name>):
 *     steel | aluminium | titanium | cfrp_ud | gfrp_ud | cfrp_quasi
 *
 * Boundary condition types (6-DOF)
 * ─────────────────────────────────
 *   clamped | pinned | simply_supported | roller_x | roller_y | roller_z
 *   symmetry_x | symmetry_y | antisymmetry_x | antisymmetry_y
 *   fixed_drill  (fix θz only, for numerical stability in flat plate models)
 *   free
 */

#include "mitcs6.h"
#include <string>
#include <vector>

// ── point load ────────────────────────────────────────────────────────────────
struct PointLoad {
    double x=0, y=0;
    double Fx=0, Fy=0, Fz=0;   // forces  [N]
    double Mx=0, My=0;          // moments [N·mm]
};

// ── edge traction ─────────────────────────────────────────────────────────────
struct EdgeLoad {
    std::string boundary;
    double Tx=0, Ty=0, Tz=0;   // [N/mm]
};

// ── BC specification ──────────────────────────────────────────────────────────
struct BCSpec {
    std::string boundary;
    std::string type;
};

// ── body force ────────────────────────────────────────────────────────────────
struct BodyForce {
    double fx=0, fy=0;   // [N/mm²]
};

// ── complete parsed problem ───────────────────────────────────────────────────
struct ProblemDef {
    double plate_w  = 300.0;
    double plate_h  = 200.0;
    double mesh_size=  10.0;
    std::vector<Cutout> cutouts;
    // material — isotropic defaults
    double E        = 210000.0;
    double nu       =      0.30;
    double thickness=     10.0;
    // laminate — populated from [ply] sections; overrides isotropic when non-empty
    std::vector<Ply> plies;
    // loads
    double    pressure = 0.0;
    BodyForce body_force;
    std::vector<PointLoad> point_loads;
    std::vector<EdgeLoad>  edge_loads;
    // BCs
    std::vector<BCSpec> bcs;
    // output
    bool        verify_geometry_only = false;
    std::string msh_file = "plate_mitc6.msh";
    std::string out_base = "plate_results";
};

// ── public API ────────────────────────────────────────────────────────────────
ProblemDef parse_input(const std::string& filename);
Config     to_config  (const ProblemDef& pd);
void       print_problem_def(const ProblemDef& pd);

void apply_problem_bcs  (const ProblemDef& pd, const Mesh& mesh,
                         std::vector<int>& fixed_dofs);
void apply_problem_loads(const ProblemDef& pd, const Mesh& mesh,
                         Eigen::VectorXd& F);

// Returns the 6-DOF bitmask for a BC type string
int bc_mask_public(const std::string& type);
