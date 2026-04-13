#pragma once
/*
 * mitcs6.h  —  MITC6 Shell FEM solver
 * Units : mm · N · MPa
 *
 * DOF layout per node: [u, v, w, θx, θy]  (indices 0–4)
 *   u, v   : in-plane translations
 *   w      : transverse deflection
 *   θx, θy : rotations about x- and y-axes
 *
 * Dependencies:
 *   Eigen  3.4+   (header-only)
 *   gmsh   4.x    (C++ API, link -lgmsh)
 *
 * Build example (single TU):
 *   g++ -O2 -std=c++17 main.cpp mitcs6.cpp -I/usr/include/eigen3 -lgmsh -o mitcs6
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

// ── convenience aliases ──────────────────────────────────────────────────────
using Mat30  = Eigen::Matrix<double, 30, 30>;
using Vec30  = Eigen::Matrix<double, 30,  1>;
using Mat3   = Eigen::Matrix<double,  3,  3>;
using Mat2   = Eigen::Matrix<double,  2,  2>;
using Mat3x30= Eigen::Matrix<double,  3, 30>;
using Mat2x30= Eigen::Matrix<double,  2, 30>;
using Vec6d  = Eigen::Matrix<double,  6,  1>;
using Vec3d  = Eigen::Matrix<double,  3,  1>;
using Vec2d  = Eigen::Matrix<double,  2,  1>;
using Coords6= Eigen::Matrix<double,  6,  2>;   // 6 nodes × (x,y)
using SpMat  = Eigen::SparseMatrix<double>;
using Trip   = Eigen::Triplet<double>;

// ── cutout descriptor ─────────────────────────────────────────────────────────
enum class CutoutType { Circle, Ellipse, Rectangle, RoundedRectangle };

struct Cutout {
    CutoutType type;
    double cx, cy;   // centre
    double w, h;     // width / height  (or diameter for circle via r)
    double r;        // corner radius (rounded_rectangle) or circle radius
};

// ── problem configuration ─────────────────────────────────────────────────────
struct Config {
    // plate
    double plate_w   = 300.0;
    double plate_h   = 200.0;
    std::vector<Cutout> cutouts;   // arbitrary number of cutouts
    double mesh_size = 10.0;
    // material
    double E         = 210000.0;
    double nu        =      0.30;
    double thickness =     10.0;
    // load
    double pressure  =     -0.1;   // lateral pressure [MPa]; –ve → downward
    // flags
    bool   verify_geometry_only = false;
    std::string msh_file = "plate_mitc6.msh";
    std::string out_base = "plate_results";
};

// ── mesh data returned from create_mesh() ────────────────────────────────────
struct Mesh {
    // node_tag → sequential index 0..N-1
    std::unordered_map<std::size_t, int> node_id;
    // node index → (x, y, z)
    std::vector<std::array<double, 3>>   coords;
    // flat connectivity: every 6 entries = one TRI6 element (gmsh node tags)
    std::vector<std::size_t>             tri6;
    int n_nodes()  const { return static_cast<int>(coords.size()); }
    int n_elems()  const { return static_cast<int>(tri6.size()) / 6; }
    int n_dof()    const { return n_nodes() * 5; }
};

// ── per-element stress-recovery result ───────────────────────────────────────
struct ElemResponse {
    Vec3d eps_m;   // membrane strains    [εxx, εyy, γxy]
    Vec3d kappa;   // curvatures          [κxx, κyy, κxy]
    Vec2d gamma;   // transverse shears   [γxz, γyz]
    Vec3d Nm;      // membrane resultants [Nx,  Ny,  Nxy]  N/mm
    Vec3d M;       // moment resultants   [Mx,  My,  Mxy]  N·mm/mm
    Vec2d Q;       // shear resultants    [Qx,  Qy]        N/mm
};

// ── surface principal stresses (top / bottom fibre) ──────────────────────────
struct SurfaceStress {
    double s1, s2, theta_deg;   // σ₁, σ₂, angle [°]
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
// ELEMENT — low-level kernels (also usable as unit-test targets)
// ============================================================
void   shape_functions (double xi, double eta, double N[6]);
void   shape_deriv     (double xi, double eta, double dN_dxi[6], double dN_deta[6]);
// Returns detJ; fills dN_dx[6], dN_dy[6], N[6]
double jacobian        (double xi, double eta, const Coords6& c,
                        double N[6], double dN_dx[6], double dN_dy[6]);
Mat30  mitc6_stiffness (const Coords6& c, double E, double nu, double t,
                        double kappa_s = 5.0/6.0, double alpha_drill = 1e-6);
ElemResponse recover_element_response(const Coords6& c, const Vec30& ue,
                                      double E, double nu, double t,
                                      double kappa_s = 5.0/6.0);

// ============================================================
// ASSEMBLY
// ============================================================
Coords6 elem_coords(const std::size_t enodes[6], const Mesh& mesh);
Vec30   elem_disps (const std::size_t enodes[6], const Mesh& mesh,
                    const Eigen::VectorXd& U);

SpMat assemble_stiffness(const Mesh& mesh, double E, double nu, double t);

Eigen::VectorXd assemble_pressure_load(const Mesh& mesh, double pressure);

std::vector<ElemResponse> recover_all(const Mesh& mesh,
                                      const Eigen::VectorXd& U,
                                      double E, double nu, double t);

// ============================================================
// BOUNDARY CONDITIONS
// ============================================================
// Returns sorted list of node tags on a named gmsh physical group (dim=1 or 2)
std::vector<std::size_t> nodes_on_group(int dim, const std::string& name);

// Appends global DOF indices to fix into `fixed`
// dof_mask: bit-field  bit0=u  bit1=v  bit2=w  bit3=θx  bit4=θy
void add_fixed_dofs(const Mesh& mesh,
                    const std::vector<std::size_t>& node_tags,
                    int dof_mask,
                    std::vector<int>& fixed);

// Convenience masks
constexpr int DOF_U   = 1 << 0;
constexpr int DOF_V   = 1 << 1;
constexpr int DOF_W   = 1 << 2;
constexpr int DOF_TX  = 1 << 3;
constexpr int DOF_TY  = 1 << 4;
constexpr int DOF_ALL = DOF_U | DOF_V | DOF_W | DOF_TX | DOF_TY;

// Apply homogeneous Dirichlet BCs: zero-out rows/cols and set diagonal=1
void apply_bcs(SpMat& K, Eigen::VectorXd& F, const std::vector<int>& fixed_dofs);

// ============================================================
// SOLVER
// ============================================================
Eigen::VectorXd solve_system(const SpMat& K, const Eigen::VectorXd& F);

// ============================================================
// POSTPROCESSING
// ============================================================
SurfaceStress surface_stress(const ElemResponse& r, double t, double z);

// Nodal averaging: assign each corner node the mean of its surrounding elements
std::vector<double> nodal_average(const Mesh& mesh,
                                  const std::vector<double>& elem_vals);

// Write VTK legacy ASCII — no external library needed, opens in ParaView
void export_vtk(const std::string& filename,
                const Mesh& mesh,
                const Eigen::VectorXd& U,
                const std::vector<ElemResponse>& resp,
                double t);

// Print per-element summary to stdout
void print_summary(const Mesh& mesh,
                   const Eigen::VectorXd& U,
                   const std::vector<ElemResponse>& resp);
