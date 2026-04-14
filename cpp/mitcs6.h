#pragma once
/*
 * mitcs6.h  —  MITC6 Shell FEM solver
 * Units : mm · N · MPa
 *
 * DOF layout per node: [u, v, w, θx, θy, θz]  (indices 0–5)
 *   u, v   : in-plane translations
 *   w      : transverse deflection
 *   θx, θy : rotations about x- and y-axes  (Reissner-Mindlin bending)
 *   θz     : drilling rotation about shell normal  (Allman/Hughes-Brezzi)
 *
 * Material models
 *   Isotropic  — single E, nu, t  →  diagonal ABD, B = 0
 *   Laminate   — N plies (E1,E2,G12,nu12,angle,tk)  →  full ABD + B coupling
 *
 * Stress recovery
 *   Gauss-point stresses at 3 integration points per element
 *   SPR (Superconvergent Patch Recovery) for smooth nodal stresses
 *
 * Dependencies
 *   Eigen 3.4+  (header-only)
 *   gmsh  4.x   (C++ API, link -lgmsh)
 *
 * Build
 *   g++ -O2 -std=c++17 main.cpp mitcs6.cpp input_parser.cpp
 *       -I/usr/include/eigen3 -lgmsh -ldl -lm -o mitcs6
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

// ── matrix/vector aliases ────────────────────────────────────────────────────
using Mat36  = Eigen::Matrix<double, 36, 36>;   // 6 DOF/node × 6 nodes
using Vec36  = Eigen::Matrix<double, 36,  1>;
using Mat3x36= Eigen::Matrix<double,  3, 36>;
using Mat2x36= Eigen::Matrix<double,  2, 36>;
using Mat3   = Eigen::Matrix<double,  3,  3>;
using Mat2   = Eigen::Matrix<double,  2,  2>;
using Vec3d  = Eigen::Matrix<double,  3,  1>;
using Vec2d  = Eigen::Matrix<double,  2,  1>;
using Coords6= Eigen::Matrix<double,  6,  2>;   // 6 nodes × (x,y)
using SpMat  = Eigen::SparseMatrix<double>;
using Trip   = Eigen::Triplet<double>;

// ── DOF masks (6-DOF per node) ───────────────────────────────────────────────
constexpr int DOF_U   = 1 << 0;
constexpr int DOF_V   = 1 << 1;
constexpr int DOF_W   = 1 << 2;
constexpr int DOF_TX  = 1 << 3;
constexpr int DOF_TY  = 1 << 4;
constexpr int DOF_TZ  = 1 << 5;   // drilling
constexpr int DOF_ALL = 0x3F;      // all six

// ── cutout ───────────────────────────────────────────────────────────────────
enum class CutoutType { Circle, Ellipse, Rectangle, RoundedRectangle };
struct Cutout {
    CutoutType type = CutoutType::Circle;
    double cx=0, cy=0, w=0, h=0, r=0;
};

// ── single composite ply ─────────────────────────────────────────────────────
struct Ply {
    double E1   = 210000.0; // fibre-dir Young's modulus [MPa]
    double E2   = 210000.0; // transverse Young's modulus [MPa]
    double G12  =  80769.0; // in-plane shear modulus [MPa]
    double nu12 =      0.30;// major Poisson's ratio
    double angle=      0.0; // fibre angle from x-axis [deg]
    double tk   =      1.0; // ply thickness [mm]
};

// ── ABD constitutive matrices (Classical Laminate Plate Theory) ───────────────
//
//  [Nm]   [A  B  0] [eps_m ]
//  [M ] = [B  D  0] [kappa ]
//  [Q ]   [0  0  S] [gamma ]
//
//  A [N/mm]     — membrane stiffness
//  B [N]        — membrane-bending coupling (0 for symmetric laminates)
//  D [N·mm]     — bending stiffness
//  S [N/mm]     — transverse shear stiffness (diagonal)
struct ABD {
    Mat3 A = Mat3::Zero();
    Mat3 B = Mat3::Zero();
    Mat3 D = Mat3::Zero();
    Mat2 S = Mat2::Zero();
};

// Build ABD from a ply stack
ABD compute_ABD(const std::vector<Ply>& plies, double kappa_s = 5.0/6.0);

// Convenience: isotropic single layer
ABD isotropic_ABD(double E, double nu, double t, double kappa_s = 5.0/6.0);

// ── problem configuration ────────────────────────────────────────────────────
struct Config {
    double plate_w   = 300.0;
    double plate_h   = 200.0;
    std::vector<Cutout> cutouts;
    double mesh_size = 10.0;
    // isotropic material (used when plies is empty)
    double E         = 210000.0;
    double nu        =      0.30;
    double thickness =     10.0;
    // laminate material (overrides E/nu/thickness when non-empty)
    std::vector<Ply> plies;
    // load
    double pressure  = 0.0;
    // flags
    bool   verify_geometry_only = false;
    std::string msh_file = "plate_mitc6.msh";
    std::string out_base = "plate_results";
    // helper: returns total laminate thickness (sum of ply tk) or thickness
    double total_thickness() const {
        if (plies.empty()) return thickness;
        double s = 0; for (auto& p : plies) s += p.tk; return s;
    }
};

// ── mesh ─────────────────────────────────────────────────────────────────────
struct Mesh {
    std::unordered_map<std::size_t, int> node_id;
    std::vector<std::array<double,3>>    coords;
    std::vector<std::size_t>             tri6;       // 6 tags per element
    std::vector<std::vector<int>>        node_elems; // built by build_adjacency
    int n_nodes() const { return (int)coords.size(); }
    int n_elems() const { return (int)tri6.size() / 6; }
    int n_dof()   const { return n_nodes() * 6; }
};

void build_adjacency(Mesh& mesh);

// ── per-element stress result ─────────────────────────────────────────────────
struct ElemResponse {
    // centroid (ξ=η=1/3) — quick check and element-level output
    Vec3d eps_m, kappa;
    Vec2d gamma;
    Vec3d Nm, M;
    Vec2d Q;
    // Gauss-point resultants (3 points) — consumed by SPR
    std::array<Vec3d,3> Nm_gp, M_gp;
    std::array<Vec2d,3> Q_gp;
    std::array<std::array<double,2>,3> gp_xy; // physical (x,y) of each GP
};

// ── SPR nodal stress ──────────────────────────────────────────────────────────
struct NodalStress {
    Vec3d Nm;   // membrane resultants [Nx,Ny,Nxy]
    Vec3d M;    // moment resultants   [Mx,My,Mxy]
    Vec2d Q;    // shear resultants    [Qx,Qy]
};

// ── surface principal stresses ────────────────────────────────────────────────
struct SurfaceStress {
    double s1, s2, theta_deg, von_mises;
};

// ============================================================
// GEOMETRY
// ============================================================
void create_geometry(const Config& cfg);

// ============================================================
// MESH
// ============================================================
Mesh create_mesh(const Config& cfg);

// ============================================================
// ELEMENT KERNELS
// ============================================================
void   shape_functions(double xi, double eta, double N[6]);
void   shape_deriv    (double xi, double eta, double dNxi[6], double dNeta[6]);
double jacobian       (double xi, double eta, const Coords6& c,
                       double N[6], double dN_dx[6], double dN_dy[6]);

// 36×36 stiffness including drilling DOF (Allman membrane enrichment)
// gamma_drill: fraction of membrane stiffness used for drilling coupling (~0.25)
Mat36 mitc6_stiffness(const Coords6& c, const ABD& abd,
                      double gamma_drill = 0.25);

ElemResponse recover_element_response(const Coords6& c, const Vec36& ue,
                                      const ABD& abd);

// ============================================================
// ASSEMBLY
// ============================================================
Coords6 elem_coords(const std::size_t enodes[6], const Mesh& mesh);
Vec36   elem_disps (const std::size_t enodes[6], const Mesh& mesh,
                    const Eigen::VectorXd& U);

SpMat           assemble_stiffness   (const Mesh& mesh, const ABD& abd);
Eigen::VectorXd assemble_pressure_load(const Mesh& mesh, double pressure);
std::vector<ElemResponse> recover_all(const Mesh& mesh,
                                      const Eigen::VectorXd& U,
                                      const ABD& abd);

// ============================================================
// SPR STRESS RECOVERY
// ============================================================
// For each node: collect Gauss-point resultants from all elements in the patch,
// fit a complete quadratic polynomial by least squares, evaluate at the node.
// Returns one NodalStress per mesh node.
std::vector<NodalStress> spr_recovery(const Mesh& mesh,
                                      const std::vector<ElemResponse>& resp);

// ============================================================
// BOUNDARY CONDITIONS & SOLVER
// ============================================================
std::vector<std::size_t> nodes_on_group(int dim, const std::string& name);
void add_fixed_dofs(const Mesh& mesh,
                    const std::vector<std::size_t>& node_tags,
                    int dof_mask,
                    std::vector<int>& fixed);
void apply_bcs(SpMat& K, Eigen::VectorXd& F, const std::vector<int>& fixed_dofs);
Eigen::VectorXd solve_system(const SpMat& K, const Eigen::VectorXd& F);

// ============================================================
// POSTPROCESSING
// ============================================================
SurfaceStress surface_stress_from_resultants(const Vec3d& Nm, const Vec3d& M,
                                             double t, double z);

std::vector<double> nodal_average(const Mesh& mesh,
                                  const std::vector<double>& elem_vals);

// export_vtk writes SPR nodal stresses — not centroid averages
void export_vtk(const std::string& filename,
                const Mesh& mesh,
                const Eigen::VectorXd& U,
                const std::vector<ElemResponse>& resp,
                const std::vector<NodalStress>& nodal,
                double t);

void print_summary(const Mesh& mesh,
                   const Eigen::VectorXd& U,
                   const std::vector<ElemResponse>& resp);
