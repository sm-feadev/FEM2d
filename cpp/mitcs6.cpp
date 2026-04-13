/*
 * mitcs6.cpp  —  MITC6 Shell FEM implementation
 * See mitcs6.h for documentation.
 */
#include "mitcs6.h"
#include <gmsh.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
// ============================================================
// GEOMETRY
// ============================================================
static int make_cutout(const Cutout& co)
{
    switch (co.type) {
    case CutoutType::Circle:
        if (co.r <= 0.0) throw std::runtime_error("Circle cutout requires r > 0");
        return gmsh::model::occ::addDisk(co.cx, co.cy, 0.0, co.r, co.r);
    case CutoutType::Ellipse:
        return gmsh::model::occ::addDisk(co.cx, co.cy, 0.0, co.w / 2.0, co.h / 2.0);
    case CutoutType::Rectangle:
        return gmsh::model::occ::addRectangle(co.cx - co.w/2.0, co.cy - co.h/2.0, 0.0, co.w, co.h);
    case CutoutType::RoundedRectangle: {
        double x0 = co.cx - co.w / 2.0;
        double y0 = co.cy - co.h / 2.0;
        double rr = std::min({co.r, 0.5*co.w, 0.5*co.h});
        // corner centres
        int c_bl = gmsh::model::occ::addPoint(x0+rr,      y0+rr,      0.0);
        int c_br = gmsh::model::occ::addPoint(x0+co.w-rr, y0+rr,      0.0);
        int c_tr = gmsh::model::occ::addPoint(x0+co.w-rr, y0+co.h-rr, 0.0);
        int c_tl = gmsh::model::occ::addPoint(x0+rr,      y0+co.h-rr, 0.0);
        // edge start/end points
        int p1 = gmsh::model::occ::addPoint(x0+rr,       y0,         0.0);
        int p2 = gmsh::model::occ::addPoint(x0+co.w-rr,  y0,         0.0);
        int p3 = gmsh::model::occ::addPoint(x0+co.w,     y0+rr,      0.0);
        int p4 = gmsh::model::occ::addPoint(x0+co.w,     y0+co.h-rr, 0.0);
        int p5 = gmsh::model::occ::addPoint(x0+co.w-rr,  y0+co.h,    0.0);
        int p6 = gmsh::model::occ::addPoint(x0+rr,       y0+co.h,    0.0);
        int p7 = gmsh::model::occ::addPoint(x0,          y0+co.h-rr, 0.0);
        int p8 = gmsh::model::occ::addPoint(x0,          y0+rr,      0.0);
        // duplicate arc-end points (OCC requires separate tags)
        int a1 = gmsh::model::occ::addPoint(x0+co.w-rr,  y0,         0.0);
        int a2 = gmsh::model::occ::addPoint(x0+co.w,     y0+rr,      0.0);
        int a3 = gmsh::model::occ::addPoint(x0+co.w,     y0+co.h-rr, 0.0);
        int a4 = gmsh::model::occ::addPoint(x0+co.w-rr,  y0+co.h,    0.0);
        int a5 = gmsh::model::occ::addPoint(x0+rr,       y0+co.h,    0.0);
        int a6 = gmsh::model::occ::addPoint(x0,          y0+co.h-rr, 0.0);
        int a7 = gmsh::model::occ::addPoint(x0,          y0+rr,      0.0);
        int a8 = gmsh::model::occ::addPoint(x0+rr,       y0,         0.0);
        int l1  = gmsh::model::occ::addLine(p1, p2);  int ar1 = gmsh::model::occ::addCircleArc(a1, c_br, a2);
        int l2  = gmsh::model::occ::addLine(p3, p4);  int ar2 = gmsh::model::occ::addCircleArc(a3, c_tr, a4);
        int l3  = gmsh::model::occ::addLine(p5, p6);  int ar3 = gmsh::model::occ::addCircleArc(a5, c_tl, a6);
        int l4  = gmsh::model::occ::addLine(p7, p8);  int ar4 = gmsh::model::occ::addCircleArc(a7, c_bl, a8);
        int loop = gmsh::model::occ::addCurveLoop({l1, ar1, l2, ar2, l3, ar3, l4, ar4});
        return gmsh::model::occ::addPlaneSurface({loop});
    }
    }
    throw std::runtime_error("Unknown CutoutType");
}
void create_geometry(const Config& cfg)
{
    gmsh::model::add("plate_with_cutouts");
    int plate = gmsh::model::occ::addRectangle(0.0, 0.0, 0.0, cfg.plate_w, cfg.plate_h);
    // build all cutout surfaces
    std::vector<std::pair<int,int>> cutout_tags;
    for (const auto& co : cfg.cutouts)
        cutout_tags.push_back({2, make_cutout(co)});
    if (!cutout_tags.empty()) {
        std::vector<std::pair<int,int>> plate_vec = {{2, plate}};
        gmsh::vectorpair out_dim_tags;
        std::vector<gmsh::vectorpair> out_dim_tags_map;
        gmsh::model::occ::cut(plate_vec, cutout_tags, out_dim_tags, out_dim_tags_map, -1, true, true);
    }
    gmsh::model::occ::synchronize();
    // ── find the remaining surface ──────────────────────────────────────────
    std::vector<std::pair<int,int>> surfs;
    gmsh::model::getEntities(surfs, 2);
    if (surfs.empty()) throw std::runtime_error("No surface after boolean cut.");
    int domain_surf = surfs[0].second;
    // ── classify boundary curves ────────────────────────────────────────────
    std::vector<std::pair<int,int>> bnd;
    gmsh::model::getBoundary({{2, domain_surf}}, bnd, false, false);
    double tol = 1e-6 * std::max(cfg.plate_w, cfg.plate_h);
    std::vector<int> left_c, right_c, bottom_c, top_c, hole_c;
    // Each hole gets its own named group HOLE_0, HOLE_1, … so individual
    // boundary conditions can be applied per cutout if needed. All hole
    // curves are also collected into a single HOLE group for convenience.
    // We identify each hole by checking which cutout centre it is closest to.
    // Pre-collect per-cutout hole curve buckets
    int nco = static_cast<int>(cfg.cutouts.size());
    std::vector<std::vector<int>> per_hole_c(nco);
    for (auto& cv : bnd) {
        int ctag = cv.second;
        double xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(1, ctag, xmin, ymin, zmin, xmax, ymax, zmax);
        double xmid = 0.5*(xmin+xmax);
        double ymid = 0.5*(ymin+ymax);
        if      (std::abs(xmid - 0.0)        <= tol) left_c.push_back(ctag);
        else if (std::abs(xmid - cfg.plate_w) <= tol) right_c.push_back(ctag);
        else if (std::abs(ymid - 0.0)         <= tol) bottom_c.push_back(ctag);
        else if (std::abs(ymid - cfg.plate_h) <= tol) top_c.push_back(ctag);
        else {
            hole_c.push_back(ctag);
            // assign to nearest cutout centre
            if (nco > 0) {
                int best = 0;
                double best_d = 1e30;
                for (int k = 0; k < nco; ++k) {
                    double dx = xmid - cfg.cutouts[k].cx;
                    double dy = ymid - cfg.cutouts[k].cy;
                    double d  = std::sqrt(dx*dx + dy*dy);
                    if (d < best_d) { best_d = d; best = k; }
                }
                per_hole_c[best].push_back(ctag);
            }
        }
    }
    // ── register physical groups ────────────────────────────────────────────
    auto reg = [&](int dim, const std::vector<int>& tags, const std::string& name){
        if (tags.empty()) throw std::runtime_error(name + " boundary not found.");
        int pg = gmsh::model::addPhysicalGroup(dim, tags);
        gmsh::model::setPhysicalName(dim, pg, name);
    };
    reg(1, left_c,   "LEFT");
    reg(1, right_c,  "RIGHT");
    reg(1, bottom_c, "BOTTOM");
    reg(1, top_c,    "TOP");
    if (!hole_c.empty()) {
        int pg = gmsh::model::addPhysicalGroup(1, hole_c);
        gmsh::model::setPhysicalName(1, pg, "HOLE");
    }
    // individual hole groups HOLE_0, HOLE_1, …
    for (int k = 0; k < nco; ++k) {
        if (!per_hole_c[k].empty()) {
            int pg = gmsh::model::addPhysicalGroup(1, per_hole_c[k]);
            gmsh::model::setPhysicalName(1, pg, "HOLE_" + std::to_string(k));
        }
    }
    {
        int pg = gmsh::model::addPhysicalGroup(2, {domain_surf});
        gmsh::model::setPhysicalName(2, pg, "DOMAIN");
    }
}
// ============================================================
// MESH
// ============================================================
Mesh create_mesh(const Config& cfg)
{
    gmsh::option::setNumber("Mesh.CharacteristicLengthMin", cfg.mesh_size);
    gmsh::option::setNumber("Mesh.CharacteristicLengthMax", cfg.mesh_size);
    gmsh::option::setNumber("Mesh.ElementOrder",            2);
    gmsh::option::setNumber("Mesh.SecondOrderLinear",       0);
    gmsh::option::setNumber("Mesh.HighOrderOptimize",       1);
    gmsh::option::setNumber("Mesh.Algorithm",               6);
    gmsh::option::setNumber("Mesh.Optimize",                1);
    gmsh::option::setNumber("Mesh.MshFileVersion",          2.2);
    gmsh::model::mesh::generate(2);
    gmsh::write(cfg.msh_file);
    // ── nodes ───────────────────────────────────────────────────────────────
    std::vector<std::size_t> ntags;
    std::vector<double>      ncoords, nparams;
    gmsh::model::mesh::getNodes(ntags, ncoords, nparams, -1, -1, false, false);
    Mesh mesh;
    mesh.node_id.reserve(ntags.size());
    mesh.coords.resize(ntags.size());
    for (std::size_t i = 0; i < ntags.size(); ++i) {
        mesh.node_id[ntags[i]] = static_cast<int>(i);
        mesh.coords[i] = {ncoords[3*i], ncoords[3*i+1], ncoords[3*i+2]};
    }
    // ── TRI6 elements (gmsh element type 9) ─────────────────────────────────
    std::vector<int>            etypes;
    std::vector<std::vector<std::size_t>> etags, econn;
    gmsh::model::mesh::getElements(etypes, etags, econn, 2, -1);
    bool found = false;
    for (std::size_t k = 0; k < etypes.size(); ++k) {
        if (etypes[k] == 9) {
            mesh.tri6 = econn[k];
            found = true;
            break;
        }
    }
    if (!found) throw std::runtime_error("No TRI6 (type-9) elements found in mesh.");
    if (mesh.tri6.size() % 6 != 0)
        throw std::runtime_error("TRI6 connectivity length not a multiple of 6.");
    return mesh;
}
// ============================================================
// ELEMENT — shape functions and kernels
// ============================================================
void shape_functions(double xi, double eta, double N[6])
{
    double L1 = xi, L2 = eta, L3 = 1.0 - xi - eta;
    N[0] = L1*(2.0*L1 - 1.0);
    N[1] = L2*(2.0*L2 - 1.0);
    N[2] = L3*(2.0*L3 - 1.0);
    N[3] = 4.0*L1*L2;
    N[4] = 4.0*L2*L3;
    N[5] = 4.0*L3*L1;
}
void shape_deriv(double xi, double eta, double dNxi[6], double dNeta[6])
{
    double L3 = 1.0 - xi - eta;
    dNxi[0] =  4.0*xi  - 1.0;   dNeta[0] = 0.0;
    dNxi[1] =  0.0;              dNeta[1] = 4.0*eta - 1.0;
    dNxi[2] =  1.0 - 4.0*L3;   dNeta[2] = 1.0 - 4.0*L3;
    dNxi[3] =  4.0*eta;         dNeta[3] = 4.0*xi;
    dNxi[4] = -4.0*eta;         dNeta[4] = 4.0*(L3 - eta);
    dNxi[5] =  4.0*(L3 - xi);   dNeta[5] = -4.0*xi;
}
double jacobian(double xi, double eta, const Coords6& c,
                double N[6], double dN_dx[6], double dN_dy[6])
{
    double dNxi[6], dNeta[6];
    shape_functions(xi, eta, N);
    shape_deriv(xi, eta, dNxi, dNeta);
    double J00=0, J01=0, J10=0, J11=0;
    for (int i = 0; i < 6; ++i) {
        J00 += dNxi[i]  * c(i,0);
        J01 += dNxi[i]  * c(i,1);
        J10 += dNeta[i] * c(i,0);
        J11 += dNeta[i] * c(i,1);
    }
    double detJ = J00*J11 - J01*J10;
    if (detJ <= 0.0) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "Non-positive Jacobian (detJ=%.3e).", detJ);
        throw std::runtime_error(buf);
    }
    double inv = 1.0 / detJ;
    double i00 =  J11*inv, i01 = -J01*inv;
    double i10 = -J10*inv, i11 =  J00*inv;
    for (int i = 0; i < 6; ++i) {
        dN_dx[i] = i00*dNxi[i] + i01*dNeta[i];
        dN_dy[i] = i10*dNxi[i] + i11*dNeta[i];
    }
    return detJ;
}
// Build Bm (3×30) and Bb (3×30) — membrane and bending B-matrices
static void bm_bb(double xi, double eta, const Coords6& c,
                  Mat3x30& Bm, Mat3x30& Bb, double& detJ)
{
    double N[6], dN_dx[6], dN_dy[6];
    detJ = jacobian(xi, eta, c, N, dN_dx, dN_dy);
    Bm.setZero(); Bb.setZero();
    for (int i = 0; i < 6; ++i) {
        int col = 5*i;
        // membrane: εxx=∂u/∂x, εyy=∂v/∂y, γxy=∂u/∂y+∂v/∂x
        Bm(0, col+0) = dN_dx[i];
        Bm(1, col+1) = dN_dy[i];
        Bm(2, col+0) = dN_dy[i];  Bm(2, col+1) = dN_dx[i];
        // bending: κxx=∂θx/∂x, κyy=∂θy/∂y, κxy=∂θx/∂y+∂θy/∂x
        Bb(0, col+3) = dN_dx[i];
        Bb(1, col+4) = dN_dy[i];
        Bb(2, col+3) = dN_dy[i];  Bb(2, col+4) = dN_dx[i];
    }
}
// Raw (un-projected) shear B-matrix Bs (2×30)
static Mat2x30 bs_raw(double xi, double eta, const Coords6& c)
{
    double N[6], dN_dx[6], dN_dy[6];
    jacobian(xi, eta, c, N, dN_dx, dN_dy);
    Mat2x30 Bs; Bs.setZero();
    for (int i = 0; i < 6; ++i) {
        int col = 5*i;
        // γxz = ∂w/∂x − θx,  γyz = ∂w/∂y − θy
        Bs(0, col+2) = dN_dx[i];  Bs(0, col+3) = -N[i];
        Bs(1, col+2) = dN_dy[i];  Bs(1, col+4) = -N[i];
    }
    return Bs;
}
// MITC6 assumed-strain shear B-matrix
static Mat2x30 bs_mitc(double xi, double eta, const Coords6& c)
{
    // tying points at edge midpoints
    static constexpr double txi[3]  = {0.5, 0.0, 0.5};
    static constexpr double teta[3] = {0.5, 0.5, 0.0};
    Mat2x30 Bs_tied[3];
    for (int a = 0; a < 3; ++a)
        Bs_tied[a] = bs_raw(txi[a], teta[a], c);
    double L1 = xi, L2 = eta, L3 = 1.0 - xi - eta;
    double ws[3] = {L1*(1.-L1), L2*(1.-L2), L3*(1.-L3)};
    double s = ws[0] + ws[1] + ws[2];
    if (std::abs(s) < 1e-14) { ws[0]=ws[1]=ws[2]=1.0/3.0; }
    else { ws[0]/=s; ws[1]/=s; ws[2]/=s; }
    Mat2x30 Bs_hat = ws[0]*Bs_tied[0] + ws[1]*Bs_tied[1] + ws[2]*Bs_tied[2];
    return Bs_hat;
}
// 3-point Gauss rule on reference triangle
static void gauss3(double gxi[3], double geta[3], double gw[3])
{
    gxi[0]=1./6.; geta[0]=1./6.; gw[0]=1./6.;
    gxi[1]=2./3.; geta[1]=1./6.; gw[1]=1./6.;
    gxi[2]=1./6.; geta[2]=2./3.; gw[2]=1./6.;
}
Mat30 mitc6_stiffness(const Coords6& c, double E, double nu, double t,
                      double kappa_s, double alpha_drill)
{
    double G     = E / (2.0*(1.0+nu));
    double fac_m = E*t       / (1.0-nu*nu);
    double fac_b = E*t*t*t   / (12.0*(1.0-nu*nu));
    Mat3 C33;
    C33 << 1., nu, 0.,
           nu, 1., 0.,
           0., 0., (1.-nu)/2.;
    Mat3  A_mat = fac_m * C33;
    Mat3  D_mat = fac_b * C33;
    Mat2  S_mat = (kappa_s * G * t) * Mat2::Identity();
    double gxi[3], geta[3], gw[3];
    gauss3(gxi, geta, gw);
    Mat30 Ke = Mat30::Zero();
    // membrane + bending
    for (int g = 0; g < 3; ++g) {
        Mat3x30 Bm, Bb; double detJ;
        bm_bb(gxi[g], geta[g], c, Bm, Bb, detJ);
        double fac = detJ * gw[g];
        Ke += fac * (Bm.transpose() * A_mat * Bm
                   + Bb.transpose() * D_mat * Bb);
    }
    // MITC shear
    for (int g = 0; g < 3; ++g) {
        double N[6], dNdx[6], dNdy[6];
        double detJ = jacobian(gxi[g], geta[g], c, N, dNdx, dNdy);
        Mat2x30 Bs = bs_mitc(gxi[g], geta[g], c);
        Ke += (detJ * gw[g]) * (Bs.transpose() * S_mat * Bs);
    }
    // drilling stabilisation
    double alpha = alpha_drill * fac_m;
    for (int i = 0; i < 6; ++i) {
        Ke(5*i+3, 5*i+3) += alpha;
        Ke(5*i+4, 5*i+4) += alpha;
    }
    return Ke;
}
ElemResponse recover_element_response(const Coords6& c, const Vec30& ue,
                                      double E, double nu, double t,
                                      double kappa_s)
{
    constexpr double XI0 = 1.0/3.0, ETA0 = 1.0/3.0;
    Mat3x30 Bm, Bb; double detJ;
    bm_bb(XI0, ETA0, c, Bm, Bb, detJ);
    Mat2x30 Bs = bs_mitc(XI0, ETA0, c);
    ElemResponse r;
    r.eps_m = Bm * ue;
    r.kappa = Bb * ue;
    r.gamma = Bs * ue;
    double G     = E / (2.0*(1.0+nu));
    double fac_m = E*t     / (1.0-nu*nu);
    double fac_b = E*t*t*t / (12.0*(1.0-nu*nu));
    Mat3 C33;
    C33 << 1., nu, 0.,  nu, 1., 0.,  0., 0., (1.-nu)/2.;
    r.Nm = fac_m * C33 * r.eps_m;
    r.M  = fac_b * C33 * r.kappa;
    r.Q  = (kappa_s * G * t) * r.gamma;
    return r;
}
// ============================================================
// ASSEMBLY HELPERS
// ============================================================
Coords6 elem_coords(const std::size_t enodes[6], const Mesh& mesh)
{
    Coords6 c;
    for (int i = 0; i < 6; ++i) {
        auto it = mesh.node_id.find(enodes[i]);
        if (it == mesh.node_id.end())
            throw std::runtime_error("elem_coords: unknown node tag");
        int idx = it->second;
        c(i,0) = mesh.coords[idx][0];
        c(i,1) = mesh.coords[idx][1];
    }
    return c;
}
Vec30 elem_disps(const std::size_t enodes[6], const Mesh& mesh,
                 const Eigen::VectorXd& U)
{
    Vec30 ue;
    for (int i = 0; i < 6; ++i) {
        int idx = mesh.node_id.at(enodes[i]);
        for (int d = 0; d < 5; ++d)
            ue(5*i+d) = U(5*idx+d);
    }
    return ue;
}
SpMat assemble_stiffness(const Mesh& mesh, double E, double nu, double t)
{
    int ndof = mesh.n_dof();
    std::vector<Trip> trips;
    trips.reserve(mesh.n_elems() * 30 * 30);
    int ne = mesh.n_elems();
    for (int e = 0; e < ne; ++e) {
        const std::size_t* en = mesh.tri6.data() + 6*e;
        Coords6 c = elem_coords(en, mesh);
        Mat30   Ke = mitc6_stiffness(c, E, nu, t);
        // global DOF indices for this element
        int gdof[30];
        for (int i = 0; i < 6; ++i) {
            int n = mesh.node_id.at(en[i]);
            for (int d = 0; d < 5; ++d)
                gdof[5*i+d] = 5*n+d;
        }
        for (int i = 0; i < 30; ++i)
            for (int j = 0; j < 30; ++j)
                trips.emplace_back(gdof[i], gdof[j], Ke(i,j));
    }
    SpMat K(ndof, ndof);
    K.setFromTriplets(trips.begin(), trips.end());
    return K;
}
Eigen::VectorXd assemble_pressure_load(const Mesh& mesh, double pressure)
{
    Eigen::VectorXd F = Eigen::VectorXd::Zero(mesh.n_dof());
    double gxi[3], geta[3], gw[3];
    gauss3(gxi, geta, gw);
    int ne = mesh.n_elems();
    for (int e = 0; e < ne; ++e) {
        const std::size_t* en = mesh.tri6.data() + 6*e;
        Coords6 c = elem_coords(en, mesh);
        Vec30 fe = Vec30::Zero();
        for (int g = 0; g < 3; ++g) {
            double N[6], dNdx[6], dNdy[6];
            double detJ = jacobian(gxi[g], geta[g], c, N, dNdx, dNdy);
            for (int i = 0; i < 6; ++i)
                fe(5*i+2) += N[i] * pressure * detJ * gw[g];  // w DOF
        }
        for (int i = 0; i < 6; ++i) {
            int n = mesh.node_id.at(en[i]);
            for (int d = 0; d < 5; ++d)
                F(5*n+d) += fe(5*i+d);
        }
    }
    return F;
}
std::vector<ElemResponse> recover_all(const Mesh& mesh,
                                      const Eigen::VectorXd& U,
                                      double E, double nu, double t)
{
    int ne = mesh.n_elems();
    std::vector<ElemResponse> resp(ne);
    for (int e = 0; e < ne; ++e) {
        const std::size_t* en = mesh.tri6.data() + 6*e;
        Coords6 c  = elem_coords(en, mesh);
        Vec30   ue = elem_disps(en, mesh, U);
        resp[e]    = recover_element_response(c, ue, E, nu, t);
    }
    return resp;
}
// ============================================================
// BOUNDARY CONDITIONS
// ============================================================
std::vector<std::size_t> nodes_on_group(int dim, const std::string& name)
{
    std::vector<std::pair<int,int>> groups;
    gmsh::model::getPhysicalGroups(groups, dim);
    for (auto& pg : groups) {
        std::string pname;
        gmsh::model::getPhysicalName(pg.first, pg.second, pname);
        if (pname != name) continue;
        std::vector<int> ents;
        gmsh::model::getEntitiesForPhysicalGroup(pg.first, pg.second, ents);
        std::unordered_set<std::size_t> tags_set;
        for (int ent : ents) {
            std::vector<std::size_t> ntags;
            std::vector<double>      ncoords, nparams;
            gmsh::model::mesh::getNodes(ntags, ncoords, nparams, dim, ent, false, false);
            tags_set.insert(ntags.begin(), ntags.end());
        }
        return std::vector<std::size_t>(tags_set.begin(), tags_set.end());
    }
    throw std::runtime_error("Physical group '" + name + "' not found.");
}
void add_fixed_dofs(const Mesh& mesh,
                    const std::vector<std::size_t>& node_tags,
                    int dof_mask,
                    std::vector<int>& fixed)
{
    for (auto tag : node_tags) {
        int n    = mesh.node_id.at(tag);
        int base = 5*n;
        for (int d = 0; d < 5; ++d)
            if (dof_mask & (1 << d))
                fixed.push_back(base + d);
    }
}
void apply_bcs(SpMat& K, Eigen::VectorXd& F, const std::vector<int>& fixed_dofs)
{
    // Sorted, unique list
    std::vector<int> dofs = fixed_dofs;
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
    // Mark rows to zero in one pass over the sparse structure
    std::set<int> fixed_set(dofs.begin(), dofs.end());
    // Make K modifiable in COO form
    K.makeCompressed();
    for (int k = 0; k < K.outerSize(); ++k) {
        for (SpMat::InnerIterator it(K, k); it; ++it) {
            int r = static_cast<int>(it.row());
            int c = static_cast<int>(it.col());
            bool r_fixed = fixed_set.count(r) > 0;
            bool c_fixed = fixed_set.count(c) > 0;
            if (r_fixed || c_fixed)
                it.valueRef() = (r == c) ? 1.0 : 0.0;
        }
    }
    for (int d : dofs)
        F(d) = 0.0;
}
// ============================================================
// SOLVER
// ============================================================
Eigen::VectorXd solve_system(const SpMat& K, const Eigen::VectorXd& F)
{
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(K);
    solver.factorize(K);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("SparseLU factorization failed.");
    Eigen::VectorXd U = solver.solve(F);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("SparseLU solve failed.");
    return U;
}
// ============================================================
// POSTPROCESSING
// ============================================================
SurfaceStress surface_stress(const ElemResponse& r, double t, double z)
{
    double Nx  = r.Nm(0), Ny  = r.Nm(1), Nxy  = r.Nm(2);
    double Mx  = r.M(0),  My  = r.M(1),  Mxy  = r.M(2);
    double sx  = Nx/t + (12.*z/(t*t*t)) * Mx;
    double sy  = Ny/t + (12.*z/(t*t*t)) * My;
    double txy = Nxy/t + (12.*z/(t*t*t)) * Mxy;
    double avg = 0.5*(sx+sy);
    double rad = std::sqrt(0.25*(sx-sy)*(sx-sy) + txy*txy);
    double th  = 0.5 * std::atan2(2.*txy, sx-sy) * 180.0/M_PI;
    return {avg+rad, avg-rad, th};
}
std::vector<double> nodal_average(const Mesh& mesh,
                                   const std::vector<double>& elem_vals)
{
    int nn = mesh.n_nodes();
    std::vector<double> vals(nn, 0.0), cnts(nn, 0.0);
    int ne = mesh.n_elems();
    for (int e = 0; e < ne; ++e) {
        const std::size_t* en = mesh.tri6.data() + 6*e;
        for (int i = 0; i < 3; ++i) {   // corner nodes only
            int idx = mesh.node_id.at(en[i]);
            vals[idx] += elem_vals[e];
            cnts[idx] += 1.0;
        }
    }
    for (int i = 0; i < nn; ++i)
        if (cnts[i] > 0.0) vals[i] /= cnts[i];
    return vals;
}
void export_vtk(const std::string& filename,
                const Mesh& mesh,
                const Eigen::VectorXd& U,
                const std::vector<ElemResponse>& resp,
                double t)
{
    std::ofstream f(filename);
    if (!f) throw std::runtime_error("Cannot open " + filename + " for writing.");
    int nn = mesh.n_nodes();
    int ne = mesh.n_elems();
    f << "# vtk DataFile Version 3.0\n";
    f << "MITC6 Shell FEM Results\n";
    f << "ASCII\n";
    f << "DATASET UNSTRUCTURED_GRID\n\n";
    // Points
    f << "POINTS " << nn << " double\n";
    for (int i = 0; i < nn; ++i)
        f << mesh.coords[i][0] << " " << mesh.coords[i][1] << " " << mesh.coords[i][2] << "\n";
    // Cells — use VTK type 22 (quadratic triangle)
    f << "\nCELLS " << ne << " " << ne*7 << "\n";
    for (int e = 0; e < ne; ++e) {
        const std::size_t* en = mesh.tri6.data() + 6*e;
        f << "6";
        for (int i = 0; i < 6; ++i)
            f << " " << mesh.node_id.at(en[i]);
        f << "\n";
    }
    f << "\nCELL_TYPES " << ne << "\n";
    for (int e = 0; e < ne; ++e) f << "22\n";
    // ── point data ───────────────────────────────────────────────────────────
    f << "\nPOINT_DATA " << nn << "\n";
    // displacement vector (u,v,w)
    f << "VECTORS displacement double\n";
    for (int i = 0; i < nn; ++i)
        f << U(5*i+0) << " " << U(5*i+1) << " " << U(5*i+2) << "\n";
    // rotations (θx, θy as a 3-vector with 0 z-component)
    f << "VECTORS rotation double\n";
    for (int i = 0; i < nn; ++i)
        f << U(5*i+3) << " " << U(5*i+4) << " 0.0\n";
    // Transverse deflection w as scalar (useful for quick colour map)
    f << "SCALARS w_deflection double 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nn; ++i) f << U(5*i+2) << "\n";
    // top and bottom principal stresses — nodal averaged
    std::vector<double> top_s1(ne), top_s2(ne), bot_s1(ne), bot_s2(ne);
    std::vector<double> Mx_e(ne), My_e(ne), Nx_e(ne), Ny_e(ne);
    std::vector<double> vm_top(ne), vm_bot(ne);
    for (int e = 0; e < ne; ++e) {
        auto st = surface_stress(resp[e], t, +0.5*t);
        auto sb = surface_stress(resp[e], t, -0.5*t);
        top_s1[e] = st.s1;  top_s2[e] = st.s2;
        bot_s1[e] = sb.s1;  bot_s2[e] = sb.s2;
        // von Mises on each surface: σ_vm = sqrt(s1²-s1*s2+s2²)
        vm_top[e] = std::sqrt(st.s1*st.s1 - st.s1*st.s2 + st.s2*st.s2);
        vm_bot[e] = std::sqrt(sb.s1*sb.s1 - sb.s1*sb.s2 + sb.s2*sb.s2);
        Mx_e[e] = resp[e].M(0);   My_e[e] = resp[e].M(1);
        Nx_e[e] = resp[e].Nm(0);  Ny_e[e] = resp[e].Nm(1);
    }
    auto write_scalar = [&](const std::string& name, const std::vector<double>& ev){
        auto nv = nodal_average(mesh, ev);
        f << "SCALARS " << name << " double 1\nLOOKUP_TABLE default\n";
        for (double v : nv) f << v << "\n";
    };
    write_scalar("top_sigma1",  top_s1);
    write_scalar("top_sigma2",  top_s2);
    write_scalar("bot_sigma1",  bot_s1);
    write_scalar("bot_sigma2",  bot_s2);
    write_scalar("von_mises_top", vm_top);
    write_scalar("von_mises_bot", vm_bot);
    write_scalar("Mx", Mx_e);
    write_scalar("My", My_e);
    write_scalar("Nx", Nx_e);
    write_scalar("Ny", Ny_e);
    std::printf("Exported %s  (%d nodes, %d elements)\n",
                filename.c_str(), nn, ne);
}
void print_summary(const Mesh& mesh,
                   const Eigen::VectorXd& U,
                   const std::vector<ElemResponse>& resp)
{
    double max_w = 0., max_u = 0., max_v = 0.;
    for (int i = 0; i < mesh.n_nodes(); ++i) {
        max_u = std::max(max_u, std::abs(U(5*i+0)));
        max_v = std::max(max_v, std::abs(U(5*i+1)));
        max_w = std::max(max_w, std::abs(U(5*i+2)));
    }
    std::printf("Displacement max |u| = %.4e mm\n", max_u);
    std::printf("             max |v| = %.4e mm\n", max_v);
    std::printf("             max |w| = %.4e mm\n", max_w);
    const auto& r0 = resp[0];
    std::printf("Element 0: Nm=[%.3e %.3e %.3e]  M=[%.3e %.3e %.3e]  Q=[%.3e %.3e]\n",
        r0.Nm(0), r0.Nm(1), r0.Nm(2),
        r0.M(0),  r0.M(1),  r0.M(2),
        r0.Q(0),  r0.Q(1));
}
