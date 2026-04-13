/*
 * input_parser.cpp  —  INI-style input file parser for MITC6 FEM solver
 */
#include "input_parser.h"
#include <gmsh.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

// ============================================================
// Internal parsing utilities
// ============================================================

static std::string trim(const std::string& s)
{
    auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

static bool starts_with(const std::string& s, char c) { return !s.empty() && s[0] == c; }

// Split "key = value" → {key, value}; returns false if no '=' found
static bool split_kv(const std::string& line, std::string& key, std::string& val)
{
    auto pos = line.find('=');
    if (pos == std::string::npos) return false;
    key = trim(line.substr(0, pos));
    val = trim(line.substr(pos + 1));
    return true;
}

static double to_double(const std::string& s, const std::string& ctx)
{
    try { return std::stod(s); }
    catch (...) {
        throw std::runtime_error("Expected number, got '" + s + "' (" + ctx + ")");
    }
}

static bool to_bool(const std::string& s)
{
    std::string l = lower(s);
    if (l=="true"||l=="yes"||l=="1"||l=="on")  return true;
    if (l=="false"||l=="no"||l=="0"||l=="off") return false;
    throw std::runtime_error("Expected boolean (true/false/yes/no), got '" + s + "'");
}

static CutoutType parse_cutout_type(const std::string& s)
{
    std::string l = lower(s);
    if (l=="circle")             return CutoutType::Circle;
    if (l=="ellipse")            return CutoutType::Ellipse;
    if (l=="rectangle")          return CutoutType::Rectangle;
    if (l=="rounded_rectangle"||
        l=="roundedrectangle")   return CutoutType::RoundedRectangle;
    throw std::runtime_error("Unknown cutout type '" + s +
        "'. Valid: circle | ellipse | rectangle | rounded_rectangle");
}

// ============================================================
// parse_input
// ============================================================

ProblemDef parse_input(const std::string& filename)
{
    std::ifstream fin(filename);
    if (!fin)
        throw std::runtime_error("Cannot open input file: " + filename);

    ProblemDef pd;
    std::string section;
    int         line_no = 0;

    // Mutable "current" objects built up across section key-value pairs
    Cutout     cur_cutout{};
    BCSpec     cur_bc{};
    PointLoad  cur_pl{};
    EdgeLoad   cur_el{};
    bool       in_cutout=false, in_bc=false, in_pl=false, in_el=false;

    // Helper: flush the current repeatable object when a new section header arrives
    auto flush = [&](){
        if (in_cutout) { pd.cutouts.push_back(cur_cutout);     cur_cutout={}; in_cutout=false; }
        if (in_bc)     { pd.bcs.push_back(cur_bc);             cur_bc={};     in_bc=false;     }
        if (in_pl)     { pd.point_loads.push_back(cur_pl);     cur_pl={};     in_pl=false;     }
        if (in_el)     { pd.edge_loads.push_back(cur_el);      cur_el={};     in_el=false;     }
    };

    std::string raw;
    while (std::getline(fin, raw)) {
        ++line_no;
        std::string line = trim(raw);
        // Skip blank lines and comments
        if (line.empty() || starts_with(line,'#') || starts_with(line,';')) continue;
        // Strip inline comments
        auto cpos = line.find('#');
        if (cpos != std::string::npos) line = trim(line.substr(0, cpos));
        if (line.empty()) continue;

        // Section header
        if (starts_with(line,'[')) {
            flush();   // save any pending repeatable object
            auto end = line.find(']');
            if (end == std::string::npos)
                throw std::runtime_error(filename+":"+std::to_string(line_no)+
                                         ": Missing ']' in section header");
            section = lower(trim(line.substr(1, end-1)));
            if (section=="cutout")     { in_cutout=true; cur_cutout={}; }
            else if (section=="bc")    { in_bc=true;     cur_bc={};     }
            else if (section=="point_load") { in_pl=true; cur_pl={};   }
            else if (section=="edge_load")  { in_el=true; cur_el={};   }
            continue;
        }

        // Key-value pair
        std::string key, val;
        if (!split_kv(line, key, val))
            throw std::runtime_error(filename+":"+std::to_string(line_no)+
                                     ": Expected 'key = value', got: "+line);
        key = lower(key);

        auto ctx = [&](){ return filename+":"+std::to_string(line_no)+" key="+key; };

        // ── dispatch by section ──────────────────────────────────────────────
        if (section=="geometry") {
            if      (key=="plate_w")   pd.plate_w   = to_double(val, ctx());
            else if (key=="plate_h")   pd.plate_h   = to_double(val, ctx());
            else if (key=="mesh_size") pd.mesh_size = to_double(val, ctx());
            else throw std::runtime_error(ctx()+": Unknown key in [geometry]");
        }
        else if (section=="material") {
            if      (key=="e"||key=="young_modulus") pd.E         = to_double(val,ctx());
            else if (key=="nu"||key=="poisson")      pd.nu        = to_double(val,ctx());
            else if (key=="thickness"||key=="t")     pd.thickness = to_double(val,ctx());
            else throw std::runtime_error(ctx()+": Unknown key in [material]");
        }
        else if (section=="loads") {
            if      (key=="pressure")   pd.pressure          = to_double(val,ctx());
            else if (key=="body_fx")    pd.body_force.fx     = to_double(val,ctx());
            else if (key=="body_fy")    pd.body_force.fy     = to_double(val,ctx());
            else throw std::runtime_error(ctx()+": Unknown key in [loads]");
        }
        else if (section=="cutout") {
            if      (key=="type") cur_cutout.type = parse_cutout_type(val);
            else if (key=="cx")   cur_cutout.cx   = to_double(val,ctx());
            else if (key=="cy")   cur_cutout.cy   = to_double(val,ctx());
            else if (key=="w")    cur_cutout.w    = to_double(val,ctx());
            else if (key=="h")    cur_cutout.h    = to_double(val,ctx());
            else if (key=="r")    cur_cutout.r    = to_double(val,ctx());
            else throw std::runtime_error(ctx()+": Unknown key in [cutout]");
        }
        else if (section=="bc") {
            if      (key=="boundary") cur_bc.boundary = val;
            else if (key=="type")     cur_bc.type      = lower(val);
            else throw std::runtime_error(ctx()+": Unknown key in [bc]");
        }
        else if (section=="point_load") {
            if      (key=="x")  cur_pl.x  = to_double(val,ctx());
            else if (key=="y")  cur_pl.y  = to_double(val,ctx());
            else if (key=="fx") cur_pl.Fx = to_double(val,ctx());
            else if (key=="fy") cur_pl.Fy = to_double(val,ctx());
            else if (key=="fz") cur_pl.Fz = to_double(val,ctx());
            else if (key=="mx") cur_pl.Mx = to_double(val,ctx());
            else if (key=="my") cur_pl.My = to_double(val,ctx());
            else throw std::runtime_error(ctx()+": Unknown key in [point_load]");
        }
        else if (section=="edge_load") {
            if      (key=="boundary") cur_el.boundary = val;
            else if (key=="tx")       cur_el.Tx = to_double(val,ctx());
            else if (key=="ty")       cur_el.Ty = to_double(val,ctx());
            else if (key=="tz")       cur_el.Tz = to_double(val,ctx());
            else throw std::runtime_error(ctx()+": Unknown key in [edge_load]");
        }
        else if (section=="output") {
            if      (key=="verify_geometry_only") pd.verify_geometry_only = to_bool(val);
            else if (key=="msh_file")             pd.msh_file             = val;
            else if (key=="out_base")             pd.out_base             = val;
            else throw std::runtime_error(ctx()+": Unknown key in [output]");
        }
        else {
            throw std::runtime_error(filename+":"+std::to_string(line_no)+
                                     ": Unknown section ["+section+"]");
        }
    }
    flush();   // save last pending object

    // ── basic validation ─────────────────────────────────────────────────────
    if (pd.plate_w <= 0) throw std::runtime_error("plate_w must be > 0");
    if (pd.plate_h <= 0) throw std::runtime_error("plate_h must be > 0");
    if (pd.mesh_size <= 0) throw std::runtime_error("mesh_size must be > 0");
    if (pd.E   <= 0) throw std::runtime_error("Young's modulus E must be > 0");
    if (pd.nu  <= 0 || pd.nu >= 0.5) throw std::runtime_error("Poisson's ratio nu must be in (0, 0.5)");
    if (pd.thickness <= 0) throw std::runtime_error("thickness must be > 0");
    for (std::size_t i = 0; i < pd.cutouts.size(); ++i) {
        const auto& co = pd.cutouts[i];
        if (co.type==CutoutType::Circle && co.r<=0)
            throw std::runtime_error("cutout["+std::to_string(i)+"]: circle requires r > 0");
    }
    for (const auto& bc : pd.bcs) {
        if (bc.boundary.empty())
            throw std::runtime_error("[bc] block missing 'boundary' key");
        if (bc.type.empty())
            throw std::runtime_error("[bc] block for boundary '"+bc.boundary+"' missing 'type' key");
    }
    for (const auto& el : pd.edge_loads) {
        if (el.boundary.empty())
            throw std::runtime_error("[edge_load] block missing 'boundary' key");
    }
    return pd;
}

// ============================================================
// to_config
// ============================================================

Config to_config(const ProblemDef& pd)
{
    Config cfg;
    cfg.plate_w             = pd.plate_w;
    cfg.plate_h             = pd.plate_h;
    cfg.cutouts             = pd.cutouts;
    cfg.mesh_size           = pd.mesh_size;
    cfg.E                   = pd.E;
    cfg.nu                  = pd.nu;
    cfg.thickness           = pd.thickness;
    cfg.pressure            = pd.pressure;
    cfg.verify_geometry_only= pd.verify_geometry_only;
    cfg.msh_file            = pd.msh_file;
    cfg.out_base            = pd.out_base;
    return cfg;
}

// ============================================================
// print_problem_def
// ============================================================

void print_problem_def(const ProblemDef& pd)
{
    std::printf("─────────────────────────────────────────\n");
    std::printf("Problem definition\n");
    std::printf("─────────────────────────────────────────\n");
    std::printf("Geometry    : %.0f × %.0f mm,  mesh %.1f mm\n",
                pd.plate_w, pd.plate_h, pd.mesh_size);
    std::printf("Cutouts     : %zu\n", pd.cutouts.size());
    static const char* ct_names[] = {"Circle","Ellipse","Rectangle","RoundedRectangle"};
    for (std::size_t i=0; i<pd.cutouts.size(); ++i) {
        const auto& co=pd.cutouts[i];
        std::printf("  [%zu] %s  cx=%.1f cy=%.1f w=%.1f h=%.1f r=%.1f\n",
                    i, ct_names[static_cast<int>(co.type)],
                    co.cx, co.cy, co.w, co.h, co.r);
    }
    std::printf("Material    : E=%.0f MPa  nu=%.3f  t=%.1f mm\n",
                pd.E, pd.nu, pd.thickness);
    if (pd.pressure != 0.0)
        std::printf("Load        : lateral pressure = %.4f MPa\n", pd.pressure);
    if (pd.body_force.fx!=0.0 || pd.body_force.fy!=0.0)
        std::printf("Load        : body force fx=%.4f fy=%.4f N/mm²\n",
                    pd.body_force.fx, pd.body_force.fy);
    for (std::size_t i=0; i<pd.point_loads.size(); ++i) {
        const auto& pl=pd.point_loads[i];
        std::printf("Point load  : @ (%.1f,%.1f)  Fx=%.2f Fy=%.2f Fz=%.2f Mx=%.2f My=%.2f\n",
                    pl.x, pl.y, pl.Fx, pl.Fy, pl.Fz, pl.Mx, pl.My);
    }
    for (std::size_t i=0; i<pd.edge_loads.size(); ++i) {
        const auto& el=pd.edge_loads[i];
        std::printf("Edge load   : boundary='%s'  Tx=%.3f Ty=%.3f Tz=%.3f N/mm\n",
                    el.boundary.c_str(), el.Tx, el.Ty, el.Tz);
    }
    std::printf("BCs         : %zu region(s)\n", pd.bcs.size());
    for (const auto& bc : pd.bcs)
        std::printf("  boundary='%s'  type='%s'\n", bc.boundary.c_str(), bc.type.c_str());
    std::printf("Output base : %s\n", pd.out_base.c_str());
    std::printf("─────────────────────────────────────────\n");
}

// ============================================================
// BC type → DOF mask
// ============================================================

static int bc_mask(const std::string& type)
{
    if (type=="clamped")           return DOF_ALL;
    if (type=="pinned")            return DOF_U|DOF_V|DOF_W;
    if (type=="simply_supported"||
        type=="roller_z")          return DOF_W;
    if (type=="roller_x")          return DOF_U;
    if (type=="roller_y")          return DOF_V;
    if (type=="symmetry_x")        return DOF_U|DOF_TY;
    if (type=="symmetry_y")        return DOF_V|DOF_TX;
    if (type=="antisymmetry_x")    return DOF_V|DOF_W|DOF_TX;
    if (type=="antisymmetry_y")    return DOF_U|DOF_W|DOF_TY;
    if (type=="free")              return 0;
    throw std::runtime_error("Unknown BC type '" + type +
        "'. Valid: clamped | pinned | simply_supported | roller_x | roller_y | "
        "roller_z | symmetry_x | symmetry_y | antisymmetry_x | antisymmetry_y | free");
}

void apply_problem_bcs(const ProblemDef& pd,
                       const Mesh& mesh,
                       std::vector<int>& fixed_dofs)
{
    for (const auto& bc : pd.bcs) {
        int mask = bc_mask(bc.type);
        if (mask == 0) continue;   // "free" — nothing to fix
        auto nodes = nodes_on_group(1, bc.boundary);
        add_fixed_dofs(mesh, nodes, mask, fixed_dofs);
    }
}

// ============================================================
// Load assembly
// ============================================================

// Find the mesh node index whose coordinates are closest to (tx, ty)
static int nearest_node(const Mesh& mesh, double tx, double ty)
{
    int    best = 0;
    double best_d = 1e300;
    for (int i = 0; i < mesh.n_nodes(); ++i) {
        double dx = mesh.coords[i][0] - tx;
        double dy = mesh.coords[i][1] - ty;
        double d  = dx*dx + dy*dy;
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

// Gauss rule on [-1,1] for a 3-node (quadratic) line element
static void gauss_line3(double pts[3], double wts[3])
{
    double sq = std::sqrt(3.0/5.0);
    pts[0]=-sq;  wts[0]=5./9.;
    pts[1]=0.0;  wts[1]=8./9.;
    pts[2]=+sq;  wts[2]=5./9.;
}

// Shape functions for a 3-node 1-D element at s ∈ [-1,1]
static void line3_shape(double s, double N[3])
{
    N[0] = 0.5*s*(s-1.);
    N[1] = 1.0 - s*s;
    N[2] = 0.5*s*(s+1.);
}

// Integrate an edge traction (force per unit length) along all LINE3 boundary
// segments of a named physical group and scatter into global force vector F.
// dof_offset: which DOF within a node (0=u, 1=v, 2=w)
// traction:   constant value [N/mm]
static void integrate_edge_traction(const Mesh& mesh,
                                    const std::string& boundary,
                                    int dof_offset,
                                    double traction,
                                    Eigen::VectorXd& F)
{
    if (traction == 0.0) return;

    // Collect all LINE3 (type 8) element connectivities on the boundary
    std::vector<std::pair<int,int>> groups;
    gmsh::model::getPhysicalGroups(groups, 1);
    for (auto& pg : groups) {
        std::string pname;
        gmsh::model::getPhysicalName(pg.first, pg.second, pname);
        if (pname != boundary) continue;
        std::vector<int> ents;
        gmsh::model::getEntitiesForPhysicalGroup(pg.first, pg.second, ents);
        for (int ent : ents) {
            std::vector<int>            etypes;
            std::vector<std::vector<std::size_t>> etags, econn;
            gmsh::model::mesh::getElements(etypes, etags, econn, 1, ent);
            for (std::size_t k=0; k<etypes.size(); ++k) {
                int npe = 0;
                if (etypes[k]==8)      npe=3;  // LINE3 (quadratic)
                else if (etypes[k]==1) npe=2;  // LINE2 (linear fallback)
                else continue;
                const auto& conn = econn[k];
                int ne = static_cast<int>(conn.size()) / npe;
                double gpts[3], gwts[3];
                gauss_line3(gpts, gwts);
                for (int e=0; e<ne; ++e) {
                    // gather node coords for this edge segment
                    std::vector<std::size_t> seg(conn.begin()+npe*e,
                                                  conn.begin()+npe*e+npe);
                    // build xy array for shape-function interpolation
                    // use indices 0,2,1 for LINE3 (gmsh: start,end,mid)
                    // or 0,1 for LINE2
                    int order = (npe==3) ? 3 : 2;
                    std::vector<std::array<double,2>> xy(order);
                    for (int i=0; i<order; ++i) {
                        // For LINE3, gmsh stores [n_start, n_end, n_mid]
                        // Map to parametric order [n0, n_mid, n1] = [0,2,1]
                        int gi = (npe==3) ? std::array<int,3>{0,2,1}[i] : i;
                        int idx = mesh.node_id.at(seg[gi]);
                        xy[i] = {mesh.coords[idx][0], mesh.coords[idx][1]};
                    }
                    int np = (npe==3) ? 3 : 2;
                    for (int g=0; g<np; ++g) {
                        double s = (np==3) ? gpts[g] : (gpts[g]*0.5);   // remap for LINE2
                        double wg= (np==3) ? gwts[g] : (gwts[g]*0.5);
                        double N[3]={0,0,0};
                        if (np==3) {
                            line3_shape(s, N);
                        } else {
                            N[0]=0.5*(1.-s); N[1]=0.5*(1.+s);
                        }
                        // Jacobian: dx/ds, dy/ds
                        // For LINE3: dN/ds = [s-0.5, -2s, s+0.5]
                        double dNds[3]={0,0,0};
                        if (np==3) {
                            dNds[0]=s-0.5; dNds[1]=-2.*s; dNds[2]=s+0.5;
                        } else {
                            dNds[0]=-0.5; dNds[1]=0.5;
                        }
                        double dxds=0, dyds=0;
                        for (int i=0; i<np; ++i) {
                            dxds += dNds[i]*xy[i][0];
                            dyds += dNds[i]*xy[i][1];
                        }
                        double jac = std::sqrt(dxds*dxds + dyds*dyds);
                        // scatter
                        for (int i=0; i<np; ++i) {
                            int gi2 = (npe==3) ? std::array<int,3>{0,2,1}[i] : i;
                            int nidx = mesh.node_id.at(seg[gi2]);
                            F(5*nidx + dof_offset) += N[i] * traction * jac * wg;
                        }
                    }
                }
            }
        }
        return;   // found the group — done
    }
    throw std::runtime_error("Physical group '" + boundary + "' not found for edge load.");
}

void apply_problem_loads(const ProblemDef& pd,
                         const Mesh& mesh,
                         Eigen::VectorXd& F)
{
    // ── lateral pressure (uniform) ────────────────────────────────────────────
    // (already assembled by assemble_pressure_load before this function is called
    //  if pd.pressure != 0; handled in main.cpp)

    // ── body force (in-plane, per unit area) ──────────────────────────────────
    if (pd.body_force.fx != 0.0 || pd.body_force.fy != 0.0) {
        // reuse gauss3 signature from mitcs6.cpp — redeclare here via extern
        // (or just inline the same values)
        double GXI[3]={1./6.,2./3.,1./6.};
        double GETA[3]={1./6.,1./6.,2./3.};
        double GW[3]={1./6.,1./6.,1./6.};
        int ne = mesh.n_elems();
        for (int e=0; e<ne; ++e) {
            const std::size_t* en = mesh.tri6.data()+6*e;
            Coords6 c = elem_coords(en, mesh);
            for (int g=0; g<3; ++g) {
                double N[6], dNdx[6], dNdy[6];
                double detJ = jacobian(GXI[g], GETA[g], c, N, dNdx, dNdy);
                for (int i=0; i<6; ++i) {
                    int nidx = mesh.node_id.at(en[i]);
                    F(5*nidx+0) += N[i] * pd.body_force.fx * detJ * GW[g];
                    F(5*nidx+1) += N[i] * pd.body_force.fy * detJ * GW[g];
                }
            }
        }
    }

    // ── point loads ──────────────────────────────────────────────────────────
    for (const auto& pl : pd.point_loads) {
        int nidx = nearest_node(mesh, pl.x, pl.y);
        double nx = mesh.coords[nidx][0];
        double ny = mesh.coords[nidx][1];
        std::printf("  Point load applied to node %d  (%.2f,%.2f) "
                    "→ nearest mesh node @ (%.3f,%.3f)\n",
                    nidx, pl.x, pl.y, nx, ny);
        F(5*nidx+0) += pl.Fx;
        F(5*nidx+1) += pl.Fy;
        F(5*nidx+2) += pl.Fz;
        F(5*nidx+3) += pl.Mx;
        F(5*nidx+4) += pl.My;
    }

    // ── edge tractions ────────────────────────────────────────────────────────
    for (const auto& el : pd.edge_loads) {
        integrate_edge_traction(mesh, el.boundary, 0, el.Tx, F);
        integrate_edge_traction(mesh, el.boundary, 1, el.Ty, F);
        integrate_edge_traction(mesh, el.boundary, 2, el.Tz, F);
    }
}

int bc_mask_public(const std::string& type) { return bc_mask(type); }
