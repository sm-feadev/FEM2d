#pragma once
/*
 * input_parser.h  —  INI-style input file reader for MITC6 FEM solver
 *
 * File format overview
 * ────────────────────
 * Lines starting with '#' or ';' are comments.
 * Blank lines are ignored.
 * Section headers: [section_name]
 * Key-value pairs: key = value
 *
 * Repeatable sections (cutout, bc, point_load, edge_load) may appear
 * multiple times; each occurrence creates one new entry.
 *
 * Full example: see plate.inp
 */

#include "mitcs6.h"
#include <string>
#include <vector>

// ── point load (concentrated force or moment at one node location) ────────────
struct PointLoad {
    double x, y;      // approximate coordinates — nearest node is found
    double Fx = 0.0;  // [N]
    double Fy = 0.0;  // [N]
    double Fz = 0.0;  // [N]
    double Mx = 0.0;  // [N·mm]
    double My = 0.0;  // [N·mm]
};

// ── distributed edge traction on a named physical boundary ───────────────────
struct EdgeLoad {
    std::string boundary;   // e.g. "RIGHT", "HOLE_0"
    double Tx = 0.0;        // in-plane traction in x [N/mm]
    double Ty = 0.0;        // in-plane traction in y [N/mm]
    double Tz = 0.0;        // transverse traction    [N/mm]
};

// ── boundary condition specification ─────────────────────────────────────────
//
//  type string       DOFs fixed
//  ─────────────     ──────────────────────────────
//  clamped           u v w θx θy  (all 5)
//  pinned            u v w
//  simply_supported  w
//  roller_x          u
//  roller_y          v
//  roller_z          w  (synonym for simply_supported)
//  symmetry_x        u θy          (plane x=const)
//  symmetry_y        v θx          (plane y=const)
//  antisymmetry_x    v w θx        (plane x=const)
//  antisymmetry_y    u w θy        (plane y=const)
//  free              (nothing — explicitly mark as free for clarity)
//
struct BCSpec {
    std::string boundary;   // named physical group, e.g. "LEFT"
    std::string type;       // one of the strings above
};

// ── body force (per unit area = force per unit volume × thickness) ────────────
struct BodyForce {
    double fx = 0.0;   // [N/mm²]
    double fy = 0.0;   // [N/mm²]
};

// ── complete parsed problem description ──────────────────────────────────────
struct ProblemDef {
    // geometry
    double plate_w   = 300.0;
    double plate_h   = 200.0;
    double mesh_size =  10.0;
    std::vector<Cutout> cutouts;

    // material
    double E         = 210000.0;
    double nu        =      0.30;
    double thickness =     10.0;

    // loads
    double               pressure   = 0.0;   // uniform lateral [MPa]
    BodyForce            body_force;
    std::vector<PointLoad> point_loads;
    std::vector<EdgeLoad>  edge_loads;

    // boundary conditions
    std::vector<BCSpec> bcs;

    // output / run control
    bool        verify_geometry_only = false;
    std::string msh_file = "plate_mitc6.msh";
    std::string out_base = "plate_results";
};

// ── public API ────────────────────────────────────────────────────────────────

// Parse an input file; throws std::runtime_error with file+line on any error.
ProblemDef parse_input(const std::string& filename);

// Convert ProblemDef → Config (for geometry/mesh pipeline)
Config to_config(const ProblemDef& pd);

// Print a human-readable summary of what was parsed
void print_problem_def(const ProblemDef& pd);

// Apply all BCs from a ProblemDef to the fixed-dof list
void apply_problem_bcs(const ProblemDef& pd,
                       const Mesh& mesh,
                       std::vector<int>& fixed_dofs);

// Assemble all loads from a ProblemDef into a global force vector
// (adds to existing F so it can be called after assemble_pressure_load)
void apply_problem_loads(const ProblemDef& pd,
                         const Mesh& mesh,
                         Eigen::VectorXd& F);

// Returns the DOF bitmask for a BC type string (same as internal bc_mask).
// Useful in main.cpp to check which DOFs a BC fixes.
int bc_mask_public(const std::string& type);
