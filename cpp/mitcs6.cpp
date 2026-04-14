/*
 * mitcs6.cpp  —  MITC6 Shell FEM implementation
 *
 * Three major additions vs previous version:
 *   1. SPR (Superconvergent Patch Recovery) nodal stress recovery
 *   2. CLPT ABD constitutive matrix for isotropic and laminate materials
 *   3. 6-DOF Allman drilling rotation (θz) enriched membrane formulation
 */
#include "mitcs6.h"
#include <gmsh.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <set>
#include <stdexcept>
#include <unordered_set>

// ============================================================
// CLPT — ABD constitutive matrices
// ============================================================

static Mat3 rotated_Q(const Ply& p)
{
    // Reduced stiffness in principal material axes
    double denom = 1.0 - p.nu12*(p.E2/p.E1)*p.nu12;
    if (denom <= 0) throw std::runtime_error("Invalid ply: nu12^2 * E2/E1 >= 1");
    double Q11 = p.E1  / denom;
    double Q22 = p.E2  / denom;
    double Q12 = p.nu12*p.E2 / denom;
    double Q66 = p.G12;

    double th  = p.angle * M_PI / 180.0;
    double c = std::cos(th), s = std::sin(th);
    double c2=c*c, s2=s*s, c4=c2*c2, s4=s2*s2, cs=c*s, c2s2=c2*s2;

    // Transform Q to laminate axes (Tsai-Pagano transformation)
    Mat3 Qbar;
    Qbar(0,0) = Q11*c4 + 2*(Q12+2*Q66)*c2s2 + Q22*s4;
    Qbar(1,1) = Q11*s4 + 2*(Q12+2*Q66)*c2s2 + Q22*c4;
    Qbar(0,1) = Qbar(1,0) = (Q11+Q22-4*Q66)*c2s2 + Q12*(c4+s4);
    Qbar(0,2) = Qbar(2,0) = (Q11-Q12-2*Q66)*c2*cs - (Q22-Q12-2*Q66)*s2*cs;
    Qbar(1,2) = Qbar(2,1) = (Q11-Q12-2*Q66)*s2*cs - (Q22-Q12-2*Q66)*c2*cs;
    Qbar(2,2) = (Q11+Q22-2*Q12-2*Q66)*c2s2 + Q66*(c4+s4);
    return Qbar;
}

ABD compute_ABD(const std::vector<Ply>& plies, double kappa_s)
{
    // z-coordinates of ply interfaces, with mid-plane at z=0
    int N = (int)plies.size();
    double total = 0; for (auto& p : plies) total += p.tk;
    std::vector<double> z(N+1);
    z[0] = -0.5*total;
    for (int k = 0; k < N; ++k) z[k+1] = z[k] + plies[k].tk;

    ABD abd;
    for (int k = 0; k < N; ++k) {
        Mat3 Qbar = rotated_Q(plies[k]);
        double dz  = z[k+1] - z[k];
        double dz2 = z[k+1]*z[k+1] - z[k]*z[k];
        double dz3 = z[k+1]*z[k+1]*z[k+1] - z[k]*z[k]*z[k];
        abd.A += Qbar * dz;
        abd.B += Qbar * 0.5*dz2;
        abd.D += Qbar * (1.0/3.0)*dz3;
    }

    // Transverse shear: Whitney-Pagano shear correction
    // Simplified: use G13, G23 averaged from plies weighted by thickness
    double G13_avg = 0, G23_avg = 0;
    for (auto& p : plies) {
        // Approximate: G13 = G12 * (E2/E1)^0.5, G23 ≈ G12 for transverse isotropy
        G13_avg += p.G12 * plies[0].tk;
        G23_avg += p.G12 * plies[0].tk;
    }
    G13_avg /= total; G23_avg /= total;
    abd.S(0,0) = kappa_s * G13_avg * total;
    abd.S(1,1) = kappa_s * G23_avg * total;
    return abd;
}

ABD isotropic_ABD(double E, double nu, double t, double kappa_s)
{
    double G = E / (2.0*(1.0+nu));
    double fac_m = E*t     / (1.0-nu*nu);
    double fac_b = E*t*t*t / (12.0*(1.0-nu*nu));
    Mat3 C33;
    C33 << 1., nu, 0., nu, 1., 0., 0., 0., (1.-nu)/2.;
    ABD abd;
    abd.A = fac_m * C33;
    abd.D = fac_b * C33;
    // B stays zero (symmetric single layer)
    abd.S(0,0) = abd.S(1,1) = kappa_s * G * t;
    return abd;
}

// ============================================================
// GEOMETRY
// ============================================================

static int make_cutout(const Cutout& co)
{
    switch (co.type) {
    case CutoutType::Circle:
        if (co.r <= 0) throw std::runtime_error("Circle cutout requires r > 0");
        return gmsh::model::occ::addDisk(co.cx, co.cy, 0.0, co.r, co.r);
    case CutoutType::Ellipse:
        return gmsh::model::occ::addDisk(co.cx, co.cy, 0.0, co.w/2, co.h/2);
    case CutoutType::Rectangle:
        return gmsh::model::occ::addRectangle(co.cx-co.w/2, co.cy-co.h/2, 0.0, co.w, co.h);
    case CutoutType::RoundedRectangle: {
        double x0=co.cx-co.w/2, y0=co.cy-co.h/2;
        double rr=std::min({co.r,0.5*co.w,0.5*co.h});
        int c_bl=gmsh::model::occ::addPoint(x0+rr,      y0+rr,      0);
        int c_br=gmsh::model::occ::addPoint(x0+co.w-rr, y0+rr,      0);
        int c_tr=gmsh::model::occ::addPoint(x0+co.w-rr, y0+co.h-rr, 0);
        int c_tl=gmsh::model::occ::addPoint(x0+rr,      y0+co.h-rr, 0);
        int p1=gmsh::model::occ::addPoint(x0+rr,      y0,       0);
        int p2=gmsh::model::occ::addPoint(x0+co.w-rr, y0,       0);
        int p3=gmsh::model::occ::addPoint(x0+co.w,    y0+rr,    0);
        int p4=gmsh::model::occ::addPoint(x0+co.w,    y0+co.h-rr,0);
        int p5=gmsh::model::occ::addPoint(x0+co.w-rr, y0+co.h,  0);
        int p6=gmsh::model::occ::addPoint(x0+rr,      y0+co.h,  0);
        int p7=gmsh::model::occ::addPoint(x0,         y0+co.h-rr,0);
        int p8=gmsh::model::occ::addPoint(x0,         y0+rr,    0);
        int a1=gmsh::model::occ::addPoint(x0+co.w-rr, y0,       0);
        int a2=gmsh::model::occ::addPoint(x0+co.w,    y0+rr,    0);
        int a3=gmsh::model::occ::addPoint(x0+co.w,    y0+co.h-rr,0);
        int a4=gmsh::model::occ::addPoint(x0+co.w-rr, y0+co.h,  0);
        int a5=gmsh::model::occ::addPoint(x0+rr,      y0+co.h,  0);
        int a6=gmsh::model::occ::addPoint(x0,         y0+co.h-rr,0);
        int a7=gmsh::model::occ::addPoint(x0,         y0+rr,    0);
        int a8=gmsh::model::occ::addPoint(x0+rr,      y0,       0);
        int l1=gmsh::model::occ::addLine(p1,p2); int ar1=gmsh::model::occ::addCircleArc(a1,c_br,a2);
        int l2=gmsh::model::occ::addLine(p3,p4); int ar2=gmsh::model::occ::addCircleArc(a3,c_tr,a4);
        int l3=gmsh::model::occ::addLine(p5,p6); int ar3=gmsh::model::occ::addCircleArc(a5,c_tl,a6);
        int l4=gmsh::model::occ::addLine(p7,p8); int ar4=gmsh::model::occ::addCircleArc(a7,c_bl,a8);
        int loop=gmsh::model::occ::addCurveLoop({l1,ar1,l2,ar2,l3,ar3,l4,ar4});
        return gmsh::model::occ::addPlaneSurface({loop});
    }
    }
    throw std::runtime_error("Unknown CutoutType");
}

void create_geometry(const Config& cfg)
{
    gmsh::model::add("plate_with_cutouts");
    int plate = gmsh::model::occ::addRectangle(0,0,0,cfg.plate_w,cfg.plate_h);
    std::vector<std::pair<int,int>> co_tags;
    for (auto& co : cfg.cutouts) co_tags.push_back({2, make_cutout(co)});
    if (!co_tags.empty()) {
        gmsh::vectorpair out_dt; std::vector<gmsh::vectorpair> out_map;
        gmsh::model::occ::cut({{2,plate}}, co_tags, out_dt, out_map, -1, true, true);
    }
    gmsh::model::occ::synchronize();

    std::vector<std::pair<int,int>> surfs;
    gmsh::model::getEntities(surfs,2);
    if (surfs.empty()) throw std::runtime_error("No surface after cut.");
    int dom = surfs[0].second;

    std::vector<std::pair<int,int>> bnd;
    gmsh::model::getBoundary({{2,dom}},bnd,false,false);

    double tol = 1e-6 * std::max(cfg.plate_w, cfg.plate_h);
    int nco = (int)cfg.cutouts.size();
    std::vector<int> lc,rc,bc,tc,hc;
    std::vector<std::vector<int>> per_hole(nco);
    for (auto& cv : bnd) {
        int ct=cv.second;
        double xn,yn,zn,xx,yx,zx;
        gmsh::model::getBoundingBox(1,ct,xn,yn,zn,xx,yx,zx);
        double xm=0.5*(xn+xx), ym=0.5*(yn+yx);
        if      (std::abs(xm-0.0)        <=tol) lc.push_back(ct);
        else if (std::abs(xm-cfg.plate_w)<=tol) rc.push_back(ct);
        else if (std::abs(ym-0.0)        <=tol) bc.push_back(ct);
        else if (std::abs(ym-cfg.plate_h)<=tol) tc.push_back(ct);
        else {
            hc.push_back(ct);
            if (nco>0) {
                int best=0; double bd=1e30;
                for (int k=0;k<nco;++k) {
                    double dx=xm-cfg.cutouts[k].cx, dy=ym-cfg.cutouts[k].cy;
                    double d=dx*dx+dy*dy;
                    if (d<bd){bd=d;best=k;}
                }
                per_hole[best].push_back(ct);
            }
        }
    }
    auto reg=[&](int dim, const std::vector<int>& tags, const std::string& nm){
        if (tags.empty()) throw std::runtime_error(nm+" boundary not found.");
        int pg=gmsh::model::addPhysicalGroup(dim,tags);
        gmsh::model::setPhysicalName(dim,pg,nm);
    };
    reg(1,lc,"LEFT"); reg(1,rc,"RIGHT"); reg(1,bc,"BOTTOM"); reg(1,tc,"TOP");
    if (!hc.empty()){int pg=gmsh::model::addPhysicalGroup(1,hc);gmsh::model::setPhysicalName(1,pg,"HOLE");}
    for (int k=0;k<nco;++k)
        if (!per_hole[k].empty()){int pg=gmsh::model::addPhysicalGroup(1,per_hole[k]);gmsh::model::setPhysicalName(1,pg,"HOLE_"+std::to_string(k));}
    {int pg=gmsh::model::addPhysicalGroup(2,{dom}); gmsh::model::setPhysicalName(2,pg,"DOMAIN");}
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

    std::vector<std::size_t> ntags; std::vector<double> nc, np;
    gmsh::model::mesh::getNodes(ntags,nc,np,-1,-1,false,false);
    Mesh mesh;
    mesh.node_id.reserve(ntags.size());
    mesh.coords.resize(ntags.size());
    for (std::size_t i=0;i<ntags.size();++i){
        mesh.node_id[ntags[i]]=(int)i;
        mesh.coords[i]={nc[3*i],nc[3*i+1],nc[3*i+2]};
    }
    std::vector<int> et; std::vector<std::vector<std::size_t>> etg,ec;
    gmsh::model::mesh::getElements(et,etg,ec,2,-1);
    bool found=false;
    for (std::size_t k=0;k<et.size();++k)
        if (et[k]==9){mesh.tri6=ec[k];found=true;break;}
    if (!found) throw std::runtime_error("No TRI6 (type 9) elements found.");
    if (mesh.tri6.size()%6!=0) throw std::runtime_error("Bad TRI6 connectivity.");
    return mesh;
}

void build_adjacency(Mesh& mesh)
{
    int nn=mesh.n_nodes(), ne=mesh.n_elems();
    mesh.node_elems.assign(nn,{});
    for (int e=0;e<ne;++e){
        const std::size_t* en=mesh.tri6.data()+6*e;
        for (int i=0;i<6;++i)
            mesh.node_elems[mesh.node_id.at(en[i])].push_back(e);
    }
}

// ============================================================
// ELEMENT KERNELS
// ============================================================

void shape_functions(double xi, double eta, double N[6])
{
    double L1=xi, L2=eta, L3=1.0-xi-eta;
    N[0]=L1*(2*L1-1); N[1]=L2*(2*L2-1); N[2]=L3*(2*L3-1);
    N[3]=4*L1*L2;     N[4]=4*L2*L3;     N[5]=4*L3*L1;
}

void shape_deriv(double xi, double eta, double dNxi[6], double dNeta[6])
{
    double L3=1.0-xi-eta;
    dNxi[0]=4*xi-1;   dNeta[0]=0;
    dNxi[1]=0;        dNeta[1]=4*eta-1;
    dNxi[2]=1-4*L3;   dNeta[2]=1-4*L3;
    dNxi[3]=4*eta;    dNeta[3]=4*xi;
    dNxi[4]=-4*eta;   dNeta[4]=4*(L3-eta);
    dNxi[5]=4*(L3-xi);dNeta[5]=-4*xi;
}

double jacobian(double xi, double eta, const Coords6& c,
                double N[6], double dN_dx[6], double dN_dy[6])
{
    double dNxi[6], dNeta[6];
    shape_functions(xi,eta,N);
    shape_deriv(xi,eta,dNxi,dNeta);
    double J00=0,J01=0,J10=0,J11=0;
    for (int i=0;i<6;++i){J00+=dNxi[i]*c(i,0);J01+=dNxi[i]*c(i,1);J10+=dNeta[i]*c(i,0);J11+=dNeta[i]*c(i,1);}
    double detJ=J00*J11-J01*J10;
    if (detJ<=0) {char b[64];std::snprintf(b,64,"detJ=%.3e",detJ);throw std::runtime_error(b);}
    double inv=1.0/detJ;
    double i00=J11*inv, i01=-J01*inv, i10=-J10*inv, i11=J00*inv;
    for (int i=0;i<6;++i){dN_dx[i]=i00*dNxi[i]+i01*dNeta[i];dN_dy[i]=i10*dNxi[i]+i11*dNeta[i];}
    return detJ;
}

// ── B-matrices (36 columns: 6 DOF/node × 6 nodes) ────────────────────────────
//
// DOF order per node: u(0) v(1) w(2) θx(3) θy(4) θz(5)
//
// Allman drilling enrichment adds θz into the membrane strain field:
//   u_enriched = Σ Nᵢ uᵢ + Σ (∂Nᵢ/∂y) * αᵢ    αᵢ = θzᵢ / 2
//   v_enriched = Σ Nᵢ vᵢ - Σ (∂Nᵢ/∂x) * αᵢ
// This couples the drilling rotation into εxx, εyy, γxy without a penalty.

static void bm_bb_drill(double xi, double eta, const Coords6& c,
                        Mat3x36& Bm, Mat3x36& Bb, double& detJ)
{
    double N[6], dN_dx[6], dN_dy[6];
    detJ = jacobian(xi, eta, c, N, dN_dx, dN_dy);
    Bm.setZero(); Bb.setZero();
    for (int i=0;i<6;++i) {
        int col=6*i;
        // membrane contributions from u(col+0), v(col+1)
        Bm(0,col+0)=dN_dx[i];
        Bm(1,col+1)=dN_dy[i];
        Bm(2,col+0)=dN_dy[i]; Bm(2,col+1)=dN_dx[i];
        // Allman drilling: θz(col+5) contributes via α = θz/2
        Bm(0,col+5)= 0.5*dN_dy[i];   //  ∂(α)/∂y → εxx  (∂u_drill/∂x = ∂(αN)/∂x ... simplified)
        Bm(1,col+5)=-0.5*dN_dx[i];   // -∂(α)/∂x → εyy
        Bm(2,col+5)= 0.5*(dN_dx[i]-dN_dy[i]); // symmetrized γxy drilling term
        // bending: θx(col+3) θy(col+4) — unchanged
        Bb(0,col+3)=dN_dx[i];
        Bb(1,col+4)=dN_dy[i];
        Bb(2,col+3)=dN_dy[i]; Bb(2,col+4)=dN_dx[i];
    }
}

// Raw shear Bs (2×36) — θz column is zero (shear unaffected by drilling)
static Mat2x36 bs_raw_36(double xi, double eta, const Coords6& c)
{
    double N[6], dN_dx[6], dN_dy[6];
    jacobian(xi,eta,c,N,dN_dx,dN_dy);
    Mat2x36 Bs; Bs.setZero();
    for (int i=0;i<6;++i) {
        int col=6*i;
        Bs(0,col+2)=dN_dx[i]; Bs(0,col+3)=-N[i];
        Bs(1,col+2)=dN_dy[i]; Bs(1,col+4)=-N[i];
    }
    return Bs;
}

// MITC6 assumed-strain shear with 36-column Bs
static Mat2x36 bs_mitc_36(double xi, double eta, const Coords6& c)
{
    static constexpr double txi[3]={0.5,0.0,0.5}, teta[3]={0.5,0.5,0.0};
    Mat2x36 Bt[3];
    for (int a=0;a<3;++a) Bt[a]=bs_raw_36(txi[a],teta[a],c);
    double L1=xi,L2=eta,L3=1-xi-eta;
    double ws[3]={L1*(1-L1),L2*(1-L2),L3*(1-L3)};
    double s=ws[0]+ws[1]+ws[2];
    if (std::abs(s)<1e-14){ws[0]=ws[1]=ws[2]=1.0/3.0;}else{ws[0]/=s;ws[1]/=s;ws[2]/=s;}
    return ws[0]*Bt[0]+ws[1]*Bt[1]+ws[2]*Bt[2];
}

static void gauss3(double gxi[3], double geta[3], double gw[3])
{
    gxi[0]=1./6.;geta[0]=1./6.;gw[0]=1./6.;
    gxi[1]=2./3.;geta[1]=1./6.;gw[1]=1./6.;
    gxi[2]=1./6.;geta[2]=2./3.;gw[2]=1./6.;
}

Mat36 mitc6_stiffness(const Coords6& c, const ABD& abd, double gamma_drill)
{
    double gxi[3],geta[3],gw[3]; gauss3(gxi,geta,gw);
    Mat36 Ke=Mat36::Zero();

    for (int g=0;g<3;++g) {
        Mat3x36 Bm,Bb; double detJ;
        bm_bb_drill(gxi[g],geta[g],c,Bm,Bb,detJ);
        double fac=detJ*gw[g];
        // [Bm.T  Bb.T] [A B; B D] [Bm; Bb]  expanded
        Ke += fac*(Bm.transpose()*abd.A*Bm
                 + Bm.transpose()*abd.B*Bb
                 + Bb.transpose()*abd.B*Bm
                 + Bb.transpose()*abd.D*Bb);
    }
    for (int g=0;g<3;++g) {
        double N[6],dNdx[6],dNdy[6];
        double detJ=jacobian(gxi[g],geta[g],c,N,dNdx,dNdy);
        Mat2x36 Bs=bs_mitc_36(gxi[g],geta[g],c);
        Ke += (detJ*gw[g])*(Bs.transpose()*abd.S*Bs);
    }

    // Drilling stabilisation: Hughes-Brezzi parameter γ_d
    // Use a fraction of the average membrane stiffness A_avg
    double A_avg = (abd.A(0,0)+abd.A(1,1))*0.5;
    double stab  = gamma_drill * A_avg;
    for (int i=0;i<6;++i) Ke(6*i+5,6*i+5) += stab;

    return Ke;
}

ElemResponse recover_element_response(const Coords6& c, const Vec36& ue,
                                      const ABD& abd)
{
    double gxi[3],geta[3],gw[3]; gauss3(gxi,geta,gw);
    ElemResponse r;

    // centroid strains and resultants
    {
        constexpr double XI0=1./3., ETA0=1./3.;
        Mat3x36 Bm,Bb; double detJ;
        bm_bb_drill(XI0,ETA0,c,Bm,Bb,detJ);
        Mat2x36 Bs=bs_mitc_36(XI0,ETA0,c);
        r.eps_m = Bm*ue;
        r.kappa = Bb*ue;
        r.gamma = Bs*ue;
        r.Nm = abd.A*r.eps_m + abd.B*r.kappa;
        r.M  = abd.B*r.eps_m + abd.D*r.kappa;
        r.Q  = abd.S*r.gamma;
    }

    // Gauss-point resultants for SPR
    for (int g=0;g<3;++g) {
        Mat3x36 Bm_g,Bb_g; double detJ_g;
        bm_bb_drill(gxi[g],geta[g],c,Bm_g,Bb_g,detJ_g);
        Mat2x36 Bs_g=bs_mitc_36(gxi[g],geta[g],c);
        Vec3d em=Bm_g*ue, ka=Bb_g*ue;
        Vec2d ga=Bs_g*ue;
        r.Nm_gp[g] = abd.A*em + abd.B*ka;
        r.M_gp[g]  = abd.B*em + abd.D*ka;
        r.Q_gp[g]  = abd.S*ga;
        // physical coordinates of this Gauss point
        double N[6],dNdx[6],dNdy[6];
        jacobian(gxi[g],geta[g],c,N,dNdx,dNdy);
        double x=0,y=0;
        for (int i=0;i<6;++i){x+=N[i]*c(i,0);y+=N[i]*c(i,1);}
        r.gp_xy[g]={x,y};
    }
    return r;
}

// ============================================================
// ASSEMBLY HELPERS
// ============================================================

Coords6 elem_coords(const std::size_t enodes[6], const Mesh& mesh)
{
    Coords6 c;
    for (int i=0;i<6;++i){int idx=mesh.node_id.at(enodes[i]);c(i,0)=mesh.coords[idx][0];c(i,1)=mesh.coords[idx][1];}
    return c;
}

Vec36 elem_disps(const std::size_t enodes[6], const Mesh& mesh, const Eigen::VectorXd& U)
{
    Vec36 ue;
    for (int i=0;i<6;++i){int idx=mesh.node_id.at(enodes[i]);for (int d=0;d<6;++d) ue(6*i+d)=U(6*idx+d);}
    return ue;
}

SpMat assemble_stiffness(const Mesh& mesh, const ABD& abd)
{
    int ndof=mesh.n_dof();
    std::vector<Trip> trips; trips.reserve(mesh.n_elems()*36*36);
    for (int e=0;e<mesh.n_elems();++e) {
        const std::size_t* en=mesh.tri6.data()+6*e;
        Coords6 c=elem_coords(en,mesh);
        Mat36 Ke=mitc6_stiffness(c,abd);
        int gdof[36];
        for (int i=0;i<6;++i){int n=mesh.node_id.at(en[i]);for (int d=0;d<6;++d) gdof[6*i+d]=6*n+d;}
        for (int i=0;i<36;++i) for (int j=0;j<36;++j) trips.emplace_back(gdof[i],gdof[j],Ke(i,j));
    }
    SpMat K(ndof,ndof);
    K.setFromTriplets(trips.begin(),trips.end());
    return K;
}

Eigen::VectorXd assemble_pressure_load(const Mesh& mesh, double pressure)
{
    Eigen::VectorXd F=Eigen::VectorXd::Zero(mesh.n_dof());
    double gxi[3],geta[3],gw[3]; gauss3(gxi,geta,gw);
    for (int e=0;e<mesh.n_elems();++e) {
        const std::size_t* en=mesh.tri6.data()+6*e;
        Coords6 c=elem_coords(en,mesh);
        for (int g=0;g<3;++g) {
            double N[6],dNdx[6],dNdy[6];
            double detJ=jacobian(gxi[g],geta[g],c,N,dNdx,dNdy);
            for (int i=0;i<6;++i) {
                int n=mesh.node_id.at(en[i]);
                F(6*n+2)+=N[i]*pressure*detJ*gw[g];   // w DOF (index 2)
            }
        }
    }
    return F;
}

std::vector<ElemResponse> recover_all(const Mesh& mesh,
                                      const Eigen::VectorXd& U,
                                      const ABD& abd)
{
    int ne=mesh.n_elems();
    std::vector<ElemResponse> resp(ne);
    for (int e=0;e<ne;++e) {
        const std::size_t* en=mesh.tri6.data()+6*e;
        resp[e]=recover_element_response(elem_coords(en,mesh),elem_disps(en,mesh,U),abd);
    }
    return resp;
}

// ============================================================
// SPR STRESS RECOVERY
// ============================================================
//
// Algorithm (Zienkiewicz & Zhu, 1992):
//   For each node i:
//     1. Collect all elements sharing node i (the patch).
//     2. For each element in the patch, read 3 Gauss-point resultants
//        and their physical (x,y) coordinates.
//     3. Fit a complete quadratic polynomial in (x,y) to the Gauss-point
//        data by least squares:
//          σ*(x,y) = a0 + a1*x + a2*y + a3*x² + a4*x*y + a5*y²
//        The system is A_ls * a = b_ls  where A_ls is the Vandermonde matrix
//        of Gauss-point coordinates and b_ls is the vector of values.
//     4. Evaluate σ*(xᵢ, yᵢ) at the node coordinates to get the SPR value.
//
// For small patches (boundary nodes with < 3 elements = < 9 data points for
// 6 unknowns), fall back to a linear fit (3 unknowns: a0,a1,a2).
//
// Each stress component (Nx, Ny, Nxy, Mx, My, Mxy, Qx, Qy) is recovered
// independently with the same polynomial.

static double eval_poly(const Eigen::VectorXd& a, double x, double y, int order)
{
    if (order==1) return a(0)+a(1)*x+a(2)*y;
    return a(0)+a(1)*x+a(2)*y+a(3)*x*x+a(4)*x*y+a(5)*y*y;
}

static double fit_and_eval(const std::vector<std::array<double,2>>& pts,
                            const std::vector<double>& vals,
                            double xn, double yn)
{
    int np=(int)pts.size();
    // choose polynomial order based on number of data points
    int order = (np >= 6) ? 2 : 1;
    int nc    = (order==2) ? 6 : 3;

    // build Vandermonde matrix A (np × nc) and rhs b (np)
    Eigen::MatrixXd A(np, nc);
    Eigen::VectorXd b(np);
    for (int i=0;i<np;++i) {
        double x=pts[i][0], y=pts[i][1];
        A(i,0)=1; A(i,1)=x; A(i,2)=y;
        if (order==2){A(i,3)=x*x; A(i,4)=x*y; A(i,5)=y*y;}
        b(i)=vals[i];
    }
    // least-squares solve via normal equations (small system, SVD safer)
    Eigen::VectorXd a = A.jacobiSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(b);
    return eval_poly(a, xn, yn, order);
}

std::vector<NodalStress> spr_recovery(const Mesh& mesh,
                                      const std::vector<ElemResponse>& resp)
{
    int nn=mesh.n_nodes();
    std::vector<NodalStress> nodal(nn);

    for (int ni=0;ni<nn;++ni) {
        const auto& patch = mesh.node_elems[ni];
        double xn=mesh.coords[ni][0], yn=mesh.coords[ni][1];

        // collect Gauss-point data from all patch elements (3 GPs each)
        std::vector<std::array<double,2>> pts;
        std::vector<double> vNx,vNy,vNxy,vMx,vMy,vMxy,vQx,vQy;
        for (int e : patch) {
            for (int g=0;g<3;++g) {
                pts.push_back(resp[e].gp_xy[g]);
                vNx.push_back(resp[e].Nm_gp[g](0));
                vNy.push_back(resp[e].Nm_gp[g](1));
                vNxy.push_back(resp[e].Nm_gp[g](2));
                vMx.push_back(resp[e].M_gp[g](0));
                vMy.push_back(resp[e].M_gp[g](1));
                vMxy.push_back(resp[e].M_gp[g](2));
                vQx.push_back(resp[e].Q_gp[g](0));
                vQy.push_back(resp[e].Q_gp[g](1));
            }
        }

        if (pts.empty()) continue;  // isolated node — leave zero

        nodal[ni].Nm(0) = fit_and_eval(pts,vNx, xn,yn);
        nodal[ni].Nm(1) = fit_and_eval(pts,vNy, xn,yn);
        nodal[ni].Nm(2) = fit_and_eval(pts,vNxy,xn,yn);
        nodal[ni].M(0)  = fit_and_eval(pts,vMx, xn,yn);
        nodal[ni].M(1)  = fit_and_eval(pts,vMy, xn,yn);
        nodal[ni].M(2)  = fit_and_eval(pts,vMxy,xn,yn);
        nodal[ni].Q(0)  = fit_and_eval(pts,vQx, xn,yn);
        nodal[ni].Q(1)  = fit_and_eval(pts,vQy, xn,yn);
    }
    return nodal;
}

// ============================================================
// BOUNDARY CONDITIONS & SOLVER
// ============================================================

std::vector<std::size_t> nodes_on_group(int dim, const std::string& name)
{
    std::vector<std::pair<int,int>> groups;
    gmsh::model::getPhysicalGroups(groups,dim);
    for (auto& pg : groups) {
        std::string pname; gmsh::model::getPhysicalName(pg.first,pg.second,pname);
        if (pname!=name) continue;
        std::vector<int> ents; gmsh::model::getEntitiesForPhysicalGroup(pg.first,pg.second,ents);
        std::unordered_set<std::size_t> ts;
        for (int ent:ents){std::vector<std::size_t> nt;std::vector<double> nc,np;gmsh::model::mesh::getNodes(nt,nc,np,dim,ent,false,false);ts.insert(nt.begin(),nt.end());}
        return std::vector<std::size_t>(ts.begin(),ts.end());
    }
    throw std::runtime_error("Physical group '"+name+"' not found.");
}

void add_fixed_dofs(const Mesh& mesh,
                    const std::vector<std::size_t>& node_tags,
                    int dof_mask,
                    std::vector<int>& fixed)
{
    for (auto tag:node_tags){int n=mesh.node_id.at(tag);for (int d=0;d<6;++d) if (dof_mask&(1<<d)) fixed.push_back(6*n+d);}
}

void apply_bcs(SpMat& K, Eigen::VectorXd& F, const std::vector<int>& fixed_dofs)
{
    std::vector<int> dofs=fixed_dofs;
    std::sort(dofs.begin(),dofs.end());
    dofs.erase(std::unique(dofs.begin(),dofs.end()),dofs.end());
    std::set<int> fs(dofs.begin(),dofs.end());
    K.makeCompressed();
    for (int k=0;k<K.outerSize();++k)
        for (SpMat::InnerIterator it(K,k);it;++it){
            bool rf=fs.count((int)it.row())>0, cf=fs.count((int)it.col())>0;
            if (rf||cf) it.valueRef()=(it.row()==it.col())?1.0:0.0;
        }
    for (int d:dofs) F(d)=0.0;
}

Eigen::VectorXd solve_system(const SpMat& K, const Eigen::VectorXd& F)
{
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(K); solver.factorize(K);
    if (solver.info()!=Eigen::Success) throw std::runtime_error("SparseLU factorization failed.");
    Eigen::VectorXd U=solver.solve(F);
    if (solver.info()!=Eigen::Success) throw std::runtime_error("SparseLU solve failed.");
    return U;
}

// ============================================================
// POSTPROCESSING
// ============================================================

SurfaceStress surface_stress_from_resultants(const Vec3d& Nm, const Vec3d& M,
                                              double t, double z)
{
    double Nx=Nm(0),Ny=Nm(1),Nxy=Nm(2);
    double Mx=M(0), My=M(1), Mxy=M(2);
    double c12=12.*z/(t*t*t);
    double sx =Nx/t+c12*Mx, sy=Ny/t+c12*My, txy=Nxy/t+c12*Mxy;
    double avg=0.5*(sx+sy);
    double rad=std::sqrt(0.25*(sx-sy)*(sx-sy)+txy*txy);
    double th =0.5*std::atan2(2.*txy,sx-sy)*180./M_PI;
    double s1=avg+rad, s2=avg-rad;
    double vm=std::sqrt(s1*s1-s1*s2+s2*s2);
    return {s1,s2,th,vm};
}

std::vector<double> nodal_average(const Mesh& mesh, const std::vector<double>& ev)
{
    int nn=mesh.n_nodes(); std::vector<double> v(nn,0),c(nn,0);
    for (int e=0;e<mesh.n_elems();++e)
        for (int i=0;i<3;++i){int idx=mesh.node_id.at(mesh.tri6[6*e+i]);v[idx]+=ev[e];c[idx]+=1;}
    for (int i=0;i<nn;++i) if (c[i]>0) v[i]/=c[i];
    return v;
}

void export_vtk(const std::string& filename,
                const Mesh& mesh,
                const Eigen::VectorXd& U,
                const std::vector<ElemResponse>& resp,
                const std::vector<NodalStress>& nodal,
                double t)
{
    std::ofstream f(filename);
    if (!f) throw std::runtime_error("Cannot open "+filename);
    int nn=mesh.n_nodes(), ne=mesh.n_elems();

    f<<"# vtk DataFile Version 3.0\nMITC6 Shell FEM Results\nASCII\nDATASET UNSTRUCTURED_GRID\n\n";
    f<<"POINTS "<<nn<<" double\n";
    for (int i=0;i<nn;++i) f<<mesh.coords[i][0]<<" "<<mesh.coords[i][1]<<" "<<mesh.coords[i][2]<<"\n";
    f<<"\nCELLS "<<ne<<" "<<ne*7<<"\n";
    for (int e=0;e<ne;++e){f<<"6";for (int i=0;i<6;++i) f<<" "<<mesh.node_id.at(mesh.tri6[6*e+i]);f<<"\n";}
    f<<"\nCELL_TYPES "<<ne<<"\n";
    for (int e=0;e<ne;++e) f<<"22\n";

    f<<"\nPOINT_DATA "<<nn<<"\n";
    // displacement (u,v,w)
    f<<"VECTORS displacement double\n";
    for (int i=0;i<nn;++i) f<<U(6*i)<<" "<<U(6*i+1)<<" "<<U(6*i+2)<<"\n";
    // rotations (θx,θy,θz)
    f<<"VECTORS rotation double\n";
    for (int i=0;i<nn;++i) f<<U(6*i+3)<<" "<<U(6*i+4)<<" "<<U(6*i+5)<<"\n";
    // transverse deflection
    f<<"SCALARS w_deflection double 1\nLOOKUP_TABLE default\n";
    for (int i=0;i<nn;++i) f<<U(6*i+2)<<"\n";

    // ── SPR nodal stress resultants ────────────────────────────────────────
    auto write_nodal=[&](const std::string& nm, auto getter){
        f<<"SCALARS "<<nm<<" double 1\nLOOKUP_TABLE default\n";
        for (int i=0;i<nn;++i) f<<getter(nodal[i])<<"\n";
    };
    write_nodal("spr_Nx",  [](const NodalStress& n){return n.Nm(0);});
    write_nodal("spr_Ny",  [](const NodalStress& n){return n.Nm(1);});
    write_nodal("spr_Nxy", [](const NodalStress& n){return n.Nm(2);});
    write_nodal("spr_Mx",  [](const NodalStress& n){return n.M(0);});
    write_nodal("spr_My",  [](const NodalStress& n){return n.M(1);});
    write_nodal("spr_Mxy", [](const NodalStress& n){return n.M(2);});
    write_nodal("spr_Qx",  [](const NodalStress& n){return n.Q(0);});
    write_nodal("spr_Qy",  [](const NodalStress& n){return n.Q(1);});

    // ── SPR surface stresses (top and bottom fibre) ────────────────────────
    // Computed from SPR-recovered Nm and M at each node
    auto write_surf=[&](const std::string& nm, double z_frac, auto pick){
        f<<"SCALARS "<<nm<<" double 1\nLOOKUP_TABLE default\n";
        for (int i=0;i<nn;++i){
            auto ss=surface_stress_from_resultants(nodal[i].Nm,nodal[i].M,t,z_frac*t);
            f<<pick(ss)<<"\n";
        }
    };
    write_surf("spr_top_sigma1",   +0.5,[](const SurfaceStress& s){return s.s1;});
    write_surf("spr_top_sigma2",   +0.5,[](const SurfaceStress& s){return s.s2;});
    write_surf("spr_top_vonMises", +0.5,[](const SurfaceStress& s){return s.von_mises;});
    write_surf("spr_bot_sigma1",   -0.5,[](const SurfaceStress& s){return s.s1;});
    write_surf("spr_bot_sigma2",   -0.5,[](const SurfaceStress& s){return s.s2;});
    write_surf("spr_bot_vonMises", -0.5,[](const SurfaceStress& s){return s.von_mises;});

    // ── centroid element data (for comparison) ─────────────────────────────
    std::vector<double> cNx(ne),cNy(ne),cMx(ne),cvm_top(ne),cvm_bot(ne);
    for (int e=0;e<ne;++e){
        cNx[e]=resp[e].Nm(0); cNy[e]=resp[e].Nm(1); cMx[e]=resp[e].M(0);
        auto st=surface_stress_from_resultants(resp[e].Nm,resp[e].M,t,+0.5*t);
        auto sb=surface_stress_from_resultants(resp[e].Nm,resp[e].M,t,-0.5*t);
        cvm_top[e]=st.von_mises; cvm_bot[e]=sb.von_mises;
    }
    f<<"\nCELL_DATA "<<ne<<"\n";
    auto write_cell=[&](const std::string& nm, const std::vector<double>& v){
        f<<"SCALARS "<<nm<<" double 1\nLOOKUP_TABLE default\n";
        for (double x:v) f<<x<<"\n";
    };
    write_cell("centroid_Nx",      cNx);
    write_cell("centroid_Ny",      cNy);
    write_cell("centroid_Mx",      cMx);
    write_cell("centroid_vm_top",  cvm_top);
    write_cell("centroid_vm_bot",  cvm_bot);

    std::printf("Exported %s  (%d nodes, %d elements)\n",filename.c_str(),nn,ne);
}

void print_summary(const Mesh& mesh, const Eigen::VectorXd& U,
                   const std::vector<ElemResponse>& resp)
{
    double mu=0,mv=0,mw=0;
    for (int i=0;i<mesh.n_nodes();++i){
        mu=std::max(mu,std::abs(U(6*i)));
        mv=std::max(mv,std::abs(U(6*i+1)));
        mw=std::max(mw,std::abs(U(6*i+2)));
    }
    std::printf("Max displacements: |u|=%.4e  |v|=%.4e  |w|=%.4e  mm\n",mu,mv,mw);
    auto& r=resp[0];
    std::printf("Element 0 centroid: Nm=[%.3e %.3e %.3e]  M=[%.3e %.3e %.3e]\n",
        r.Nm(0),r.Nm(1),r.Nm(2),r.M(0),r.M(1),r.M(2));
}
