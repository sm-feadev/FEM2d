/*
 * input_parser.cpp  —  INI-style input file parser
 */
#include "input_parser.h"
#include <gmsh.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

// ============================================================
// Internal utilities
// ============================================================

static std::string trim(const std::string& s)
{
    auto a=s.find_first_not_of(" \t\r\n");
    if (a==std::string::npos) return {};
    return s.substr(a, s.find_last_not_of(" \t\r\n")-a+1);
}

static std::string lower(std::string s)
{
    std::transform(s.begin(),s.end(),s.begin(),[](unsigned char c){return std::tolower(c);});
    return s;
}

static bool split_kv(const std::string& line, std::string& key, std::string& val)
{
    auto pos=line.find('=');
    if (pos==std::string::npos) return false;
    key=trim(line.substr(0,pos)); val=trim(line.substr(pos+1)); return true;
}

static double to_double(const std::string& s, const std::string& ctx)
{
    try { return std::stod(s); }
    catch (...) { throw std::runtime_error("Expected number, got '"+s+"' ("+ctx+")"); }
}

static bool to_bool(const std::string& s)
{
    std::string l=lower(s);
    if (l=="true"||l=="yes"||l=="1"||l=="on")  return true;
    if (l=="false"||l=="no"||l=="0"||l=="off") return false;
    throw std::runtime_error("Expected boolean, got '"+s+"'");
}

static CutoutType parse_cutout_type(const std::string& s)
{
    std::string l=lower(s);
    if (l=="circle")                        return CutoutType::Circle;
    if (l=="ellipse")                       return CutoutType::Ellipse;
    if (l=="rectangle")                     return CutoutType::Rectangle;
    if (l=="rounded_rectangle"||l=="roundedrectangle") return CutoutType::RoundedRectangle;
    throw std::runtime_error("Unknown cutout type '"+s+"'");
}

// ── built-in material presets ─────────────────────────────────────────────────
static bool apply_preset(const std::string& name, ProblemDef& pd)
{
    std::string n=lower(name);
    // isotropic presets → set E, nu (thickness stays as user specified)
    if (n=="steel")     {pd.E=210000; pd.nu=0.30; return true;}
    if (n=="aluminium"||n=="aluminum") {pd.E=70000; pd.nu=0.33; return true;}
    if (n=="titanium")  {pd.E=114000; pd.nu=0.34; return true;}
    if (n=="copper")    {pd.E=120000; pd.nu=0.34; return true;}
    // laminate presets — append plies to pd.plies
    // Unidirectional CFRP (T300/N5208-style)
    if (n=="cfrp_ud") {
        pd.plies.push_back({181000,10300,7170,0.28,0.0,pd.thickness});
        return true;
    }
    // Unidirectional GFRP (E-glass/epoxy)
    if (n=="gfrp_ud") {
        pd.plies.push_back({38600,8270,4140,0.26,0.0,pd.thickness});
        return true;
    }
    // CFRP quasi-isotropic [0/+45/-45/90]s — 8 plies, equal thickness
    if (n=="cfrp_quasi") {
        double tk=pd.thickness/8.0;
        double angles[]={0,45,-45,90,90,-45,45,0};
        for (double a : angles)
            pd.plies.push_back({181000,10300,7170,0.28,a,tk});
        return true;
    }
    return false;
}

// ============================================================
// parse_input
// ============================================================

ProblemDef parse_input(const std::string& filename)
{
    std::ifstream fin(filename);
    if (!fin) throw std::runtime_error("Cannot open input file: "+filename);

    ProblemDef pd;
    std::string section;
    int line_no=0;

    Cutout  cur_co{};  bool in_co=false;
    Ply     cur_ply{}; bool in_ply=false;
    BCSpec  cur_bc{};  bool in_bc=false;
    PointLoad cur_pl{};bool in_pl=false;
    EdgeLoad  cur_el{};bool in_el=false;

    auto flush=[&](){
        if (in_co) {pd.cutouts.push_back(cur_co);    cur_co={};  in_co=false;}
        if (in_ply){pd.plies.push_back(cur_ply);     cur_ply={}; in_ply=false;}
        if (in_bc) {pd.bcs.push_back(cur_bc);        cur_bc={};  in_bc=false;}
        if (in_pl) {pd.point_loads.push_back(cur_pl);cur_pl={};  in_pl=false;}
        if (in_el) {pd.edge_loads.push_back(cur_el); cur_el={};  in_el=false;}
    };

    std::string raw;
    while (std::getline(fin,raw)) {
        ++line_no;
        std::string line=trim(raw);
        if (line.empty()||line[0]=='#'||line[0]==';') continue;
        auto cp=line.find('#');
        if (cp!=std::string::npos) line=trim(line.substr(0,cp));
        if (line.empty()) continue;

        if (line[0]=='[') {
            flush();
            auto end=line.find(']');
            if (end==std::string::npos)
                throw std::runtime_error(filename+":"+std::to_string(line_no)+": Missing ']'");
            section=lower(trim(line.substr(1,end-1)));
            if      (section=="cutout")     {in_co=true;  cur_co={};}
            else if (section=="ply")        {in_ply=true; cur_ply={};}
            else if (section=="bc")         {in_bc=true;  cur_bc={};}
            else if (section=="point_load") {in_pl=true;  cur_pl={};}
            else if (section=="edge_load")  {in_el=true;  cur_el={};}
            continue;
        }

        std::string key,val;
        if (!split_kv(line,key,val))
            throw std::runtime_error(filename+":"+std::to_string(line_no)+": Bad key=value: "+line);
        key=lower(key);
        auto ctx=[&]{return filename+":"+std::to_string(line_no)+" key="+key;};

        if (section=="geometry") {
            if      (key=="plate_w")   pd.plate_w   =to_double(val,ctx());
            else if (key=="plate_h")   pd.plate_h   =to_double(val,ctx());
            else if (key=="mesh_size") pd.mesh_size =to_double(val,ctx());
            else throw std::runtime_error(ctx()+": Unknown key in [geometry]");
        }
        else if (section=="material") {
            if      (key=="e"||key=="young_modulus") pd.E        =to_double(val,ctx());
            else if (key=="nu"||key=="poisson")      pd.nu       =to_double(val,ctx());
            else if (key=="thickness"||key=="t")     pd.thickness=to_double(val,ctx());
            else if (key=="preset") {
                if (!apply_preset(val,pd))
                    throw std::runtime_error(ctx()+": Unknown material preset '"+val+"'.\n"
                        "  Valid: steel | aluminium | titanium | copper | "
                        "cfrp_ud | gfrp_ud | cfrp_quasi");
            }
            else throw std::runtime_error(ctx()+": Unknown key in [material]");
        }
        else if (section=="ply") {
            if      (key=="e1"||key=="e_fibre")   cur_ply.E1   =to_double(val,ctx());
            else if (key=="e2"||key=="e_trans")   cur_ply.E2   =to_double(val,ctx());
            else if (key=="g12"||key=="g_shear")  cur_ply.G12  =to_double(val,ctx());
            else if (key=="nu12"||key=="poisson") cur_ply.nu12 =to_double(val,ctx());
            else if (key=="angle"||key=="theta")  cur_ply.angle=to_double(val,ctx());
            else if (key=="tk"||key=="thickness") cur_ply.tk   =to_double(val,ctx());
            else if (key=="preset") {
                // ply-level preset: e.g. preset = cfrp_t300
                std::string n=lower(val);
                if      (n=="cfrp_t300") {cur_ply.E1=181000;cur_ply.E2=10300;cur_ply.G12=7170;cur_ply.nu12=0.28;}
                else if (n=="cfrp_im7")  {cur_ply.E1=165000;cur_ply.E2= 8970;cur_ply.G12=5600;cur_ply.nu12=0.34;}
                else if (n=="gfrp_eglass"){cur_ply.E1= 38600;cur_ply.E2= 8270;cur_ply.G12=4140;cur_ply.nu12=0.26;}
                else throw std::runtime_error(ctx()+": Unknown ply preset '"+val+"'.\n"
                        "  Valid: cfrp_t300 | cfrp_im7 | gfrp_eglass");
            }
            else throw std::runtime_error(ctx()+": Unknown key in [ply]");
        }
        else if (section=="loads") {
            if      (key=="pressure")  pd.pressure        =to_double(val,ctx());
            else if (key=="body_fx")   pd.body_force.fx   =to_double(val,ctx());
            else if (key=="body_fy")   pd.body_force.fy   =to_double(val,ctx());
            else throw std::runtime_error(ctx()+": Unknown key in [loads]");
        }
        else if (section=="cutout") {
            if      (key=="type") cur_co.type=parse_cutout_type(val);
            else if (key=="cx")   cur_co.cx  =to_double(val,ctx());
            else if (key=="cy")   cur_co.cy  =to_double(val,ctx());
            else if (key=="w")    cur_co.w   =to_double(val,ctx());
            else if (key=="h")    cur_co.h   =to_double(val,ctx());
            else if (key=="r")    cur_co.r   =to_double(val,ctx());
            else throw std::runtime_error(ctx()+": Unknown key in [cutout]");
        }
        else if (section=="bc") {
            if      (key=="boundary") cur_bc.boundary=val;
            else if (key=="type")     cur_bc.type    =lower(val);
            else throw std::runtime_error(ctx()+": Unknown key in [bc]");
        }
        else if (section=="point_load") {
            if      (key=="x")  cur_pl.x =to_double(val,ctx());
            else if (key=="y")  cur_pl.y =to_double(val,ctx());
            else if (key=="fx") cur_pl.Fx=to_double(val,ctx());
            else if (key=="fy") cur_pl.Fy=to_double(val,ctx());
            else if (key=="fz") cur_pl.Fz=to_double(val,ctx());
            else if (key=="mx") cur_pl.Mx=to_double(val,ctx());
            else if (key=="my") cur_pl.My=to_double(val,ctx());
            else throw std::runtime_error(ctx()+": Unknown key in [point_load]");
        }
        else if (section=="edge_load") {
            if      (key=="boundary") cur_el.boundary=val;
            else if (key=="tx")       cur_el.Tx=to_double(val,ctx());
            else if (key=="ty")       cur_el.Ty=to_double(val,ctx());
            else if (key=="tz")       cur_el.Tz=to_double(val,ctx());
            else throw std::runtime_error(ctx()+": Unknown key in [edge_load]");
        }
        else if (section=="output") {
            if      (key=="verify_geometry_only") pd.verify_geometry_only=to_bool(val);
            else if (key=="msh_file")             pd.msh_file=val;
            else if (key=="out_base")             pd.out_base=val;
            else throw std::runtime_error(ctx()+": Unknown key in [output]");
        }
        else throw std::runtime_error(filename+":"+std::to_string(line_no)+": Unknown section ["+section+"]");
    }
    flush();

    // validation
    if (pd.plate_w<=0) throw std::runtime_error("plate_w must be > 0");
    if (pd.plate_h<=0) throw std::runtime_error("plate_h must be > 0");
    if (pd.mesh_size<=0) throw std::runtime_error("mesh_size must be > 0");
    if (pd.plies.empty()) {
        if (pd.E<=0)  throw std::runtime_error("E must be > 0");
        if (pd.nu<=0||pd.nu>=0.5) throw std::runtime_error("nu must be in (0,0.5)");
        if (pd.thickness<=0) throw std::runtime_error("thickness must be > 0");
    } else {
        for (auto& p : pd.plies)
            if (p.tk<=0) throw std::runtime_error("All ply thicknesses must be > 0");
    }
    for (auto& bc : pd.bcs) {
        if (bc.boundary.empty()) throw std::runtime_error("[bc] missing 'boundary'");
        if (bc.type.empty())     throw std::runtime_error("[bc] missing 'type'");
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
    cfg.plies               = pd.plies;
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
    std::printf("─────────────────────────────────────────────────\n");
    std::printf("Problem definition\n");
    std::printf("─────────────────────────────────────────────────\n");
    std::printf("Geometry   : %.0f × %.0f mm,  mesh %.1f mm\n",pd.plate_w,pd.plate_h,pd.mesh_size);
    std::printf("Cutouts    : %zu\n",pd.cutouts.size());
    static const char* ctn[]={"Circle","Ellipse","Rectangle","RoundedRectangle"};
    for (std::size_t i=0;i<pd.cutouts.size();++i){
        auto& co=pd.cutouts[i];
        std::printf("  [%zu] %s  cx=%.1f cy=%.1f w=%.1f h=%.1f r=%.1f\n",
                    i,ctn[(int)co.type],co.cx,co.cy,co.w,co.h,co.r);
    }
    if (pd.plies.empty()) {
        std::printf("Material   : Isotropic  E=%.0f MPa  nu=%.3f  t=%.1f mm\n",
                    pd.E,pd.nu,pd.thickness);
    } else {
        double t_total=0; for (auto& p:pd.plies) t_total+=p.tk;
        std::printf("Material   : Laminate  %zu plies  total t=%.2f mm\n",
                    pd.plies.size(),t_total);
        for (std::size_t k=0;k<pd.plies.size();++k) {
            auto& p=pd.plies[k];
            std::printf("  ply[%zu] E1=%.0f E2=%.0f G12=%.0f nu12=%.3f angle=%.1f tk=%.3f\n",
                        k,p.E1,p.E2,p.G12,p.nu12,p.angle,p.tk);
        }
    }
    if (pd.pressure!=0) std::printf("Load       : pressure = %.4f MPa\n",pd.pressure);
    if (pd.body_force.fx||pd.body_force.fy)
        std::printf("Load       : body force fx=%.4f fy=%.4f N/mm²\n",pd.body_force.fx,pd.body_force.fy);
    for (auto& pl:pd.point_loads)
        std::printf("Point load : (%.1f,%.1f)  Fx=%.1f Fy=%.1f Fz=%.1f\n",pl.x,pl.y,pl.Fx,pl.Fy,pl.Fz);
    for (auto& el:pd.edge_loads)
        std::printf("Edge load  : '%s'  Tx=%.3f Ty=%.3f Tz=%.3f N/mm\n",el.boundary.c_str(),el.Tx,el.Ty,el.Tz);
    std::printf("BCs        : %zu region(s)\n",pd.bcs.size());
    for (auto& bc:pd.bcs) std::printf("  '%s' → %s\n",bc.boundary.c_str(),bc.type.c_str());
    std::printf("─────────────────────────────────────────────────\n");
}

// ============================================================
// BC type → 6-DOF bitmask
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
    if (type=="fixed_drill")       return DOF_TZ;
    if (type=="free")              return 0;
    throw std::runtime_error("Unknown BC type '"+type+
        "'. Valid: clamped | pinned | simply_supported | roller_x | roller_y | "
        "roller_z | symmetry_x | symmetry_y | antisymmetry_x | antisymmetry_y | "
        "fixed_drill | free");
}

int bc_mask_public(const std::string& type) { return bc_mask(type); }

void apply_problem_bcs(const ProblemDef& pd, const Mesh& mesh,
                       std::vector<int>& fixed_dofs)
{
    for (auto& bc : pd.bcs) {
        int mask=bc_mask(bc.type);
        if (mask==0) continue;
        auto nodes=nodes_on_group(1,bc.boundary);
        add_fixed_dofs(mesh,nodes,mask,fixed_dofs);
    }
}

// ============================================================
// Load assembly
// ============================================================

static int nearest_node(const Mesh& mesh, double tx, double ty)
{
    int best=0; double bd=1e300;
    for (int i=0;i<mesh.n_nodes();++i) {
        double dx=mesh.coords[i][0]-tx, dy=mesh.coords[i][1]-ty;
        double d=dx*dx+dy*dy;
        if (d<bd){bd=d;best=i;}
    }
    return best;
}

static void gauss_line3(double pts[3], double wts[3])
{
    double sq=std::sqrt(3.0/5.0);
    pts[0]=-sq;wts[0]=5./9.;pts[1]=0;wts[1]=8./9.;pts[2]=sq;wts[2]=5./9.;
}

static void line3_shape(double s, double N[3])
{N[0]=0.5*s*(s-1);N[1]=1-s*s;N[2]=0.5*s*(s+1);}

static void integrate_edge_traction(const Mesh& mesh, const std::string& boundary,
                                     int dof_offset, double traction,
                                     Eigen::VectorXd& F)
{
    if (traction==0.0) return;
    std::vector<std::pair<int,int>> groups;
    gmsh::model::getPhysicalGroups(groups,1);
    for (auto& pg:groups) {
        std::string pname; gmsh::model::getPhysicalName(pg.first,pg.second,pname);
        if (pname!=boundary) continue;
        std::vector<int> ents; gmsh::model::getEntitiesForPhysicalGroup(pg.first,pg.second,ents);
        for (int ent:ents) {
            std::vector<int> et; std::vector<std::vector<std::size_t>> etg,ec;
            gmsh::model::mesh::getElements(et,etg,ec,1,ent);
            for (std::size_t k=0;k<et.size();++k) {
                int npe=(et[k]==8)?3:(et[k]==1)?2:0;
                if (!npe) continue;
                auto& conn=ec[k];
                int ne2=(int)conn.size()/npe;
                double gpts[3],gwts[3]; gauss_line3(gpts,gwts);
                for (int e=0;e<ne2;++e) {
                    std::vector<std::size_t> seg(conn.begin()+npe*e,conn.begin()+npe*e+npe);
                    int np2=(npe==3)?3:2;
                    std::vector<std::array<double,2>> xy(np2);
                    for (int i=0;i<np2;++i){
                        int gi=(npe==3)?std::array<int,3>{0,2,1}[i]:i;
                        int idx=mesh.node_id.at(seg[gi]);
                        xy[i]={mesh.coords[idx][0],mesh.coords[idx][1]};
                    }
                    for (int g=0;g<np2;++g){
                        double s=gpts[g], N[3]={0,0,0};
                        if (np2==3) line3_shape(s,N); else{N[0]=0.5*(1-s);N[1]=0.5*(1+s);}
                        double dNds[3]={s-0.5,-2*s,s+0.5};
                        if (np2==2){dNds[0]=-0.5;dNds[1]=0.5;}
                        double dxds=0,dyds=0;
                        for (int i=0;i<np2;++i){dxds+=dNds[i]*xy[i][0];dyds+=dNds[i]*xy[i][1];}
                        double jac=std::sqrt(dxds*dxds+dyds*dyds);
                        for (int i=0;i<np2;++i){
                            int gi=(npe==3)?std::array<int,3>{0,2,1}[i]:i;
                            int nidx=mesh.node_id.at(seg[gi]);
                            F(6*nidx+dof_offset)+=N[i]*traction*jac*gwts[g];
                        }
                    }
                }
            }
        }
        return;
    }
    throw std::runtime_error("Physical group '"+boundary+"' not found for edge load.");
}

void apply_problem_loads(const ProblemDef& pd, const Mesh& mesh, Eigen::VectorXd& F)
{
    // body force (in-plane, acts on u and v DOFs)
    if (pd.body_force.fx||pd.body_force.fy) {
        double GXI[3]={1./6.,2./3.,1./6.}, GETA[3]={1./6.,1./6.,2./3.}, GW[3]={1./6.,1./6.,1./6.};
        for (int e=0;e<mesh.n_elems();++e) {
            const std::size_t* en=mesh.tri6.data()+6*e;
            Coords6 c=elem_coords(en,mesh);
            for (int g=0;g<3;++g){
                double N[6],dNdx[6],dNdy[6];
                double detJ=jacobian(GXI[g],GETA[g],c,N,dNdx,dNdy);
                for (int i=0;i<6;++i){
                    int nidx=mesh.node_id.at(en[i]);
                    F(6*nidx+0)+=N[i]*pd.body_force.fx*detJ*GW[g];
                    F(6*nidx+1)+=N[i]*pd.body_force.fy*detJ*GW[g];
                }
            }
        }
    }
    // point loads
    for (auto& pl:pd.point_loads){
        int ni=nearest_node(mesh,pl.x,pl.y);
        std::printf("  Point load → node %d @ (%.2f,%.2f)\n",ni,mesh.coords[ni][0],mesh.coords[ni][1]);
        F(6*ni+0)+=pl.Fx; F(6*ni+1)+=pl.Fy; F(6*ni+2)+=pl.Fz;
        F(6*ni+3)+=pl.Mx; F(6*ni+4)+=pl.My;
    }
    // edge tractions
    for (auto& el:pd.edge_loads){
        integrate_edge_traction(mesh,el.boundary,0,el.Tx,F);
        integrate_edge_traction(mesh,el.boundary,1,el.Ty,F);
        integrate_edge_traction(mesh,el.boundary,2,el.Tz,F);
    }
}
