"""
MITC6 Shell FEM — plate with cutout
Units: mm · N · MPa

DOF layout per node: [u, v, w, θx, θy]  (indices 0–4)
  u, v  : in-plane translations
  w     : transverse (out-of-plane) deflection
  θx,θy : rotations about x and y

Supported cutouts : circle | ellipse | rectangle | rounded_rectangle
Supported BCs     : clamped edge (all 5 DOFs fixed)
Load              : uniform lateral pressure on the plate surface (→ w DOF)
"""

import gmsh
import numpy as np
import matplotlib.pyplot as plt
import meshio
from scipy.sparse import lil_matrix, csr_matrix, identity
from scipy.sparse.linalg import spsolve
from matplotlib.tri import Triangulation

# ============================================================
# CONFIG
# ============================================================

VERIFY_GEOMETRY_ONLY = False   # True → stop after geometry check

# Units: mm – N – MPa
PLATE_W  = 300.0
PLATE_H  = 200.0

CUTOUT_TYPE   = "rounded_rectangle"   # circle | ellipse | rectangle | rounded_rectangle
CX, CY        = 150.0, 100.0          # cutout centre
CUTOUT_W      = 120.0                 # cutout width  (ellipse: 2*a)
CUTOUT_H      =  60.0                 # cutout height (ellipse: 2*b)
CUTOUT_R      =  10.0                 # corner radius (rounded_rectangle) or circle radius

MESH_SIZE = 10.0

E         = 210000.0   # Young's modulus [MPa]
NU        =      0.30  # Poisson's ratio
THICKNESS =     10.0   # plate thickness [mm]
PRESSURE  =     -0.1   # lateral pressure [MPa]; negative → downward (−z)

# ============================================================
# GEOMETRY
# ============================================================

def create_geometry(plate_w, plate_h, cutout_type, cx, cy, w, h, r):
    """Build plate-with-cutout surface in gmsh OCC and tag physical groups."""
    gmsh.model.add("plate_with_cutout")
    occ = gmsh.model.occ

    plate = occ.addRectangle(0.0, 0.0, 0.0, plate_w, plate_h)

    if cutout_type == "circle":
        if r <= 0.0:
            raise ValueError("Circle cutout requires r > 0")
        cutout = occ.addDisk(cx, cy, 0.0, r, r)
    elif cutout_type == "ellipse":
        cutout = occ.addDisk(cx, cy, 0.0, w / 2.0, h / 2.0)
    elif cutout_type == "rectangle":
        cutout = occ.addRectangle(cx - w / 2.0, cy - h / 2.0, 0.0, w, h)
    elif cutout_type == "rounded_rectangle":
        x0  = cx - w / 2.0
        y0  = cy - h / 2.0
        rr  = min(r, 0.5 * w, 0.5 * h)
        # corner centres
        c_bl = occ.addPoint(x0 + rr,     y0 + rr,     0.0)
        c_br = occ.addPoint(x0 + w - rr, y0 + rr,     0.0)
        c_tr = occ.addPoint(x0 + w - rr, y0 + h - rr, 0.0)
        c_tl = occ.addPoint(x0 + rr,     y0 + h - rr, 0.0)
        # edge mid-points (start/end of arcs and lines)
        p1 = occ.addPoint(x0 + rr,         y0,           0.0)
        p2 = occ.addPoint(x0 + w - rr,     y0,           0.0)
        p3 = occ.addPoint(x0 + w,          y0 + rr,      0.0)
        p4 = occ.addPoint(x0 + w,          y0 + h - rr,  0.0)
        p5 = occ.addPoint(x0 + w - rr,     y0 + h,       0.0)
        p6 = occ.addPoint(x0 + rr,         y0 + h,       0.0)
        p7 = occ.addPoint(x0,              y0 + h - rr,  0.0)
        p8 = occ.addPoint(x0,              y0 + rr,      0.0)
        # duplicate arc-end points (OCC needs explicit separate point tags)
        a1 = occ.addPoint(x0 + w - rr,     y0,           0.0)
        a2 = occ.addPoint(x0 + w,          y0 + rr,      0.0)
        a3 = occ.addPoint(x0 + w,          y0 + h - rr,  0.0)
        a4 = occ.addPoint(x0 + w - rr,     y0 + h,       0.0)
        a5 = occ.addPoint(x0 + rr,         y0 + h,       0.0)
        a6 = occ.addPoint(x0,              y0 + h - rr,  0.0)
        a7 = occ.addPoint(x0,              y0 + rr,      0.0)
        a8 = occ.addPoint(x0 + rr,         y0,           0.0)
        l1  = occ.addLine(p1, p2)
        ar1 = occ.addCircleArc(a1, c_br, a2)
        l2  = occ.addLine(p3, p4)
        ar2 = occ.addCircleArc(a3, c_tr, a4)
        l3  = occ.addLine(p5, p6)
        ar3 = occ.addCircleArc(a5, c_tl, a6)
        l4  = occ.addLine(p7, p8)
        ar4 = occ.addCircleArc(a7, c_bl, a8)
        loop   = occ.addCurveLoop([l1, ar1, l2, ar2, l3, ar3, l4, ar4])
        cutout = occ.addPlaneSurface([loop])
    else:
        raise ValueError("cutout_type must be: circle | ellipse | rectangle | rounded_rectangle")

    occ.cut([(2, plate)], [(2, cutout)], removeObject=True, removeTool=True)
    occ.synchronize()

    surfaces = gmsh.model.getEntities(2)
    if not surfaces:
        raise RuntimeError("No surface after boolean cut.")
    domain_surf = surfaces[0][1]

    final_boundary  = gmsh.model.getBoundary([(2, domain_surf)], oriented=False, recursive=False)
    final_curve_tags = [c[1] for c in final_boundary]
    if not final_curve_tags:
        raise RuntimeError("No boundary curves found after cut.")

    tol = 1e-6 * max(plate_w, plate_h)
    left_curves, right_curves, bottom_curves, top_curves, hole_curves = [], [], [], [], []

    for ctag in final_curve_tags:
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, ctag)
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        if   abs(xmid - 0.0)      <= tol: left_curves.append(ctag)
        elif abs(xmid - plate_w)  <= tol: right_curves.append(ctag)
        elif abs(ymid - 0.0)      <= tol: bottom_curves.append(ctag)
        elif abs(ymid - plate_h)  <= tol: top_curves.append(ctag)
        else:                             hole_curves.append(ctag)

    for name, curves in [("LEFT", left_curves), ("RIGHT", right_curves),
                         ("BOTTOM", bottom_curves), ("TOP", top_curves),
                         ("HOLE", hole_curves)]:
        if not curves:
            raise RuntimeError(f"{name} boundary not found.")
        pg = gmsh.model.addPhysicalGroup(1, curves)
        gmsh.model.setPhysicalName(1, pg, name)

    pg_domain = gmsh.model.addPhysicalGroup(2, [domain_surf])
    gmsh.model.setPhysicalName(2, pg_domain, "DOMAIN")


def debug_print_gmsh_groups():
    print("\n=== Physical Groups ===")
    for dim, tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, tag)
        ents = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        print(f"  dim={dim} tag={tag} name='{name}' entities={list(ents)}")
    print("=== Curve BBoxes ===")
    for _, ctag in gmsh.model.getEntities(1):
        print(f"  curve {ctag}: {gmsh.model.getBoundingBox(1, ctag)}")
    print()

# ============================================================
# MESH
# ============================================================

def create_tri6_mesh(mesh_size=10.0, msh_file="plate_mitc6.msh", plot=True):
    """Generate second-order triangular (TRI6) mesh, return node data and connectivity."""
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin",  mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",  mesh_size)
    gmsh.option.setNumber("Mesh.ElementOrder",             2)
    gmsh.option.setNumber("Mesh.SecondOrderLinear",        0)
    gmsh.option.setNumber("Mesh.HighOrderOptimize",        1)
    gmsh.option.setNumber("Mesh.Algorithm",                6)
    gmsh.option.setNumber("Mesh.Optimize",                 1)
    gmsh.option.setNumber("Mesh.MshFileVersion",           2.2)
    gmsh.model.mesh.generate(2)
    gmsh.write(msh_file)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords_dict = {
        tag: (node_coords[3 * i], node_coords[3 * i + 1], node_coords[3 * i + 2])
        for i, tag in enumerate(node_tags)
    }

    tri6_conn = None
    for etype, _, conn in zip(*gmsh.model.mesh.getElements(dim=2)):
        if etype == 9:   # TRI6
            tri6_conn = conn
            break
    if tri6_conn is None:
        raise RuntimeError("No TRI6 elements found in mesh.")

    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(0, len(tri6_conn), 6):
            n1, n2, n3, n4, n5, n6 = tri6_conn[i:i + 6]
            p = [coords_dict[n] for n in (n1, n2, n3, n4, n5, n6)]
            ax.plot([p[0][0], p[3][0], p[1][0]], [p[0][1], p[3][1], p[1][1]], "b-", lw=0.7)
            ax.plot([p[1][0], p[4][0], p[2][0]], [p[1][1], p[4][1], p[2][1]], "b-", lw=0.7)
            ax.plot([p[2][0], p[5][0], p[0][0]], [p[2][1], p[5][1], p[0][1]], "b-", lw=0.7)
            ax.plot([p[0][0], p[1][0], p[2][0]], [p[0][1], p[1][1], p[2][1]], "ro", ms=3)
            ax.plot([p[3][0], p[4][0], p[5][0]], [p[3][1], p[4][1], p[5][1]], "go", ms=4)
        ax.set_aspect("equal"); ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_title("MITC6 Mesh"); ax.grid(True, ls="--", alpha=0.4)
        plt.tight_layout(); plt.show()

    return node_tags, coords_dict, tri6_conn

# ============================================================
# ELEMENT — MITC6 shell (Reissner–Mindlin, 5 DOF/node)
# ============================================================
# DOF order per node: u(0), v(1), w(2), θx(3), θy(4)
# Membrane  strains : [εxx, εyy, γxy]  ← u,v
# Bending curvatures: [κxx, κyy, κxy]  ← θx,θy
# Transverse shears : [γxz, γyz]       ← w,θx,θy  (MITC-projected)

def _shape_functions(xi, eta):
    L1, L2, L3 = xi, eta, 1.0 - xi - eta
    return np.array([
        L1 * (2.0 * L1 - 1.0),
        L2 * (2.0 * L2 - 1.0),
        L3 * (2.0 * L3 - 1.0),
        4.0 * L1 * L2,
        4.0 * L2 * L3,
        4.0 * L3 * L1,
    ], dtype=float)

def _shape_deriv(xi, eta):
    L3 = 1.0 - xi - eta
    dN_dxi  = np.array([4.*xi - 1.,        0.,   1. - 4.*L3,  4.*eta, -4.*eta, 4.*(L3 - xi)],  dtype=float)
    dN_deta = np.array([0.,        4.*eta - 1.,   1. - 4.*L3,  4.*xi,  4.*(L3 - eta), -4.*xi], dtype=float)
    return dN_dxi, dN_deta

def _gauss3():
    """3-point Gauss quadrature on the reference triangle (exact for cubics)."""
    return [(1./6., 1./6., 1./6.), (2./3., 1./6., 1./6.), (1./6., 2./3., 1./6.)]

def _jacobian(xi, eta, coords):
    dN_dxi, dN_deta = _shape_deriv(xi, eta)
    x, y = coords[:, 0], coords[:, 1]
    J = np.array([[dN_dxi @ x,  dN_dxi @ y],
                  [dN_deta @ x, dN_deta @ y]], dtype=float)
    detJ = np.linalg.det(J)
    if detJ <= 0.0:
        raise ValueError(f"Non-positive Jacobian (detJ={detJ:.3e}).")
    invJ = np.linalg.inv(J)
    N    = _shape_functions(xi, eta)
    dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    return N, dN_dx, dN_dy, detJ

def _bm_bb(xi, eta, coords):
    """Membrane Bm (3×30) and bending Bb (3×30) strain-displacement matrices."""
    _, dN_dx, dN_dy, detJ = _jacobian(xi, eta, coords)
    Bm = np.zeros((3, 30)); Bb = np.zeros((3, 30))
    for i in range(6):
        c = 5 * i
        # membrane: εxx=∂u/∂x, εyy=∂v/∂y, γxy=∂u/∂y+∂v/∂x
        Bm[0, c]     = dN_dx[i]
        Bm[1, c + 1] = dN_dy[i]
        Bm[2, c]     = dN_dy[i];  Bm[2, c + 1] = dN_dx[i]
        # bending: κxx=∂θx/∂x, κyy=∂θy/∂y, κxy=∂θx/∂y+∂θy/∂x
        Bb[0, c + 3] = dN_dx[i]
        Bb[1, c + 4] = dN_dy[i]
        Bb[2, c + 3] = dN_dy[i];  Bb[2, c + 4] = dN_dx[i]
    return Bm, Bb, detJ

def _bs_raw(xi, eta, coords):
    """Raw (un-projected) shear strain-displacement matrix Bs (2×30)."""
    N, dN_dx, dN_dy, _ = _jacobian(xi, eta, coords)
    Bs = np.zeros((2, 30))
    for i in range(6):
        c = 5 * i
        # γxz = ∂w/∂x − θx,  γyz = ∂w/∂y − θy
        Bs[0, c + 2] = dN_dx[i];  Bs[0, c + 3] = -N[i]
        Bs[1, c + 2] = dN_dy[i];  Bs[1, c + 4] = -N[i]
    return Bs

def _bs_mitc(xi, eta, coords):
    """MITC6 assumed-strain shear matrix: interpolate from 3 edge tying points."""
    # Tying points at edge midpoints (in area coordinates)
    tying = [(0.5, 0.5), (0.0, 0.5), (0.5, 0.0)]
    Bs_tied = [_bs_raw(txi, teta, coords) for txi, teta in tying]

    L1, L2, L3 = xi, eta, 1.0 - xi - eta
    ws = np.array([L1*(1.-L1), L2*(1.-L2), L3*(1.-L3)], dtype=float)
    s  = ws.sum()
    ws = ws / s if abs(s) > 1e-14 else np.full(3, 1.0/3.0)

    Bs_hat = sum(ws[a] * Bs_tied[a] for a in range(3))
    _, _, _, detJ = _jacobian(xi, eta, coords)
    return Bs_hat, detJ

def mitc6_stiffness(coords, E, nu, t, kappa_s=5./6., alpha_drill=1e-6):
    """
    Compute 30×30 element stiffness matrix for one MITC6 shell element.

    alpha_drill: small drilling-DOF stabilisation — set proportional to
                 the membrane stiffness magnitude (E*t) to avoid ill-conditioning.
    """
    coords = np.asarray(coords, dtype=float)
    G = E / (2.0 * (1.0 + nu))
    fac_m = E * t        / (1.0 - nu**2)
    fac_b = E * t**3     / (12.0 * (1.0 - nu**2))
    C33   = np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., (1.-nu)/2.]], dtype=float)
    A_mat = fac_m * C33          # membrane
    D_mat = fac_b * C33          # bending
    S_mat = kappa_s * G * t * np.eye(2)  # shear

    Ke = np.zeros((30, 30))
    for xi, eta, w in _gauss3():
        Bm, Bb, detJ = _bm_bb(xi, eta, coords)
        Ke += (Bm.T @ A_mat @ Bm + Bb.T @ D_mat @ Bb) * detJ * w

    for xi, eta, w in _gauss3():
        Bs_hat, detJ = _bs_mitc(xi, eta, coords)
        Ke += (Bs_hat.T @ S_mat @ Bs_hat) * detJ * w

    # Drilling stabilisation: penalise θx and θy at each node with a small
    # fraction of the membrane stiffness so the system is non-singular even
    # for fully-clamped shells where rotational DOFs are constrained.
    alpha = alpha_drill * fac_m
    for i in range(6):
        c = 5 * i
        Ke[c + 3, c + 3] += alpha
        Ke[c + 4, c + 4] += alpha
    return Ke

def recover_element_response(coords, ue, E, nu, t, kappa_s=5./6.):
    """
    Recover strains, curvatures, shear strains, and stress resultants at the
    element centroid (ξ=η=1/3).

    Returns dict:
        eps_m : membrane strains    [εxx, εyy, γxy]
        kappa : curvatures          [κxx, κyy, κxy]
        gamma : transverse shears   [γxz, γyz]
        Nm    : membrane resultants [Nx, Ny, Nxy]  [N/mm]
        M     : moment resultants   [Mx, My, Mxy]  [N·mm/mm]
        Q     : shear resultants    [Qx, Qy]       [N/mm]
    """
    coords = np.asarray(coords, dtype=float)
    ue     = np.asarray(ue, dtype=float)
    xi0, eta0 = 1./3., 1./3.

    Bm, Bb, _  = _bm_bb(xi0, eta0, coords)
    Bs_hat, _  = _bs_mitc(xi0, eta0, coords)

    eps_m = Bm    @ ue
    kappa = Bb    @ ue
    gamma = Bs_hat @ ue

    G     = E / (2.0 * (1.0 + nu))
    fac_m = E * t     / (1.0 - nu**2)
    fac_b = E * t**3  / (12.0 * (1.0 - nu**2))
    C33   = np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., (1.-nu)/2.]], dtype=float)

    Nm = fac_m * C33 @ eps_m
    M  = fac_b * C33 @ kappa
    Q  = kappa_s * G * t * gamma
    return {"eps_m": eps_m, "kappa": kappa, "gamma": gamma, "Nm": Nm, "M": M, "Q": Q}

# ============================================================
# ASSEMBLY
# ============================================================

def _elem_coords(elem_node_tags, coords_dict):
    c = np.zeros((6, 2))
    for i, tag in enumerate(elem_node_tags):
        c[i, 0], c[i, 1] = coords_dict[tag][:2]
    return c

def _elem_dofs(elem_node_tags, node_id_map, ndof=5):
    dofs = []
    for tag in elem_node_tags:
        base = node_id_map[tag] * ndof
        dofs.extend(range(base, base + ndof))
    return dofs

def assemble_stiffness(node_tags, coords_dict, tri6_conn, E, nu, t):
    node_id_map = {tag: i for i, tag in enumerate(node_tags)}
    ndof_total  = len(node_tags) * 5
    K = lil_matrix((ndof_total, ndof_total))
    nelem = len(tri6_conn) // 6
    for e in range(nelem):
        enodes  = tri6_conn[6*e:6*e + 6]
        coords  = _elem_coords(enodes, coords_dict)
        Ke      = mitc6_stiffness(coords, E, nu, t)
        edofs   = _elem_dofs(enodes, node_id_map)
        for i, I in enumerate(edofs):
            for j, J in enumerate(edofs):
                K[I, J] += Ke[i, j]
    return K.tocsr(), node_id_map

def assemble_pressure_load(node_tags, coords_dict, tri6_conn, node_id_map, pressure):
    """Consistent nodal load vector for uniform lateral pressure (acts on w DOF)."""
    F     = np.zeros(len(node_tags) * 5)
    nelem = len(tri6_conn) // 6
    for e in range(nelem):
        enodes = tri6_conn[6*e:6*e + 6]
        coords = _elem_coords(enodes, coords_dict)
        fe     = np.zeros(30)
        for xi, eta, wgt in _gauss3():
            N = _shape_functions(xi, eta)
            _, _, _, detJ = _jacobian(xi, eta, coords)
            for i in range(6):
                fe[5*i + 2] += N[i] * pressure * detJ * wgt   # w DOF only
        edofs = _elem_dofs(enodes, node_id_map)
        for i_loc, gdof in enumerate(edofs):
            F[gdof] += fe[i_loc]
    return F

def recover_all_elements(node_tags, coords_dict, tri6_conn, node_id_map, U, E, nu, t):
    responses = []
    nelem = len(tri6_conn) // 6
    for e in range(nelem):
        enodes = tri6_conn[6*e:6*e + 6]
        coords = _elem_coords(enodes, coords_dict)
        edofs  = _elem_dofs(enodes, node_id_map)
        ue     = U[edofs]
        responses.append(recover_element_response(coords, ue, E, nu, t))
    return responses

# ============================================================
# BOUNDARY CONDITIONS & SOLVER
# ============================================================

def get_nodes_on_physical_group(dim, phys_name):
    for d, tag in gmsh.model.getPhysicalGroups(dim):
        if gmsh.model.getPhysicalName(d, tag) == phys_name:
            node_tags = set()
            for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                ntags, _, _ = gmsh.model.mesh.getNodes(dim, ent)
                node_tags.update(ntags.tolist())
            return sorted(node_tags)
    raise RuntimeError(f"Physical group '{phys_name}' not found.")

def fixed_dofs_for_nodes(node_id_map, node_tags, dof_indices=(0, 1, 2, 3, 4)):
    """Return sorted list of global DOF indices to fix for the given nodes."""
    dofs = []
    for tag in node_tags:
        base = node_id_map[tag] * 5
        dofs.extend(base + np.array(dof_indices, dtype=int))
    return sorted(set(dofs))

def apply_bcs(K, F, fixed_dofs, prescribed_values=None):
    """Partition K and F into free/fixed sets, return reduced system."""
    if not isinstance(K, csr_matrix):
        raise TypeError("K must be csr_matrix")
    ndof       = K.shape[0]
    fixed_dofs = np.array(sorted(set(fixed_dofs)), dtype=int)
    free_dofs  = np.setdiff1d(np.arange(ndof), fixed_dofs)
    u_presc    = np.zeros(ndof)
    if prescribed_values:
        for dof, val in prescribed_values.items():
            u_presc[int(dof)] = float(val)
    K_ff = K[free_dofs[:, None], free_dofs]
    K_fc = K[free_dofs[:, None], fixed_dofs]
    F_f  = F[free_dofs] - K_fc @ u_presc[fixed_dofs]
    return K_ff, F_f, free_dofs, fixed_dofs, u_presc

def solve(K_ff, F_f, free_dofs, fixed_dofs, u_presc, ndof_total):
    U              = np.zeros(ndof_total)
    U[fixed_dofs]  = u_presc[fixed_dofs]
    U[free_dofs]   = spsolve(K_ff, F_f)
    return U

# ============================================================
# POSTPROCESSING
# ============================================================

def principal_stresses(sx, sy, txy):
    avg   = 0.5 * (sx + sy)
    rad   = np.sqrt((0.5 * (sx - sy))**2 + txy**2)
    theta = np.degrees(0.5 * np.arctan2(2.0 * txy, sx - sy))
    return avg + rad, avg - rad, theta

def surface_stresses(responses, t):
    """
    Recover top/bottom surface principal stresses from shell resultants.
    σ = N/t ± (6/t²) M   at z = ±t/2
    """
    nelem = len(responses)
    top_s1 = np.zeros(nelem); top_s2 = np.zeros(nelem); top_th = np.zeros(nelem)
    bot_s1 = np.zeros(nelem); bot_s2 = np.zeros(nelem); bot_th = np.zeros(nelem)
    for e, res in enumerate(responses):
        Nx, Ny, Nxy = res["Nm"]
        Mx, My, Mxy = res["M"]
        for sign, s1_arr, s2_arr, th_arr in [
                (+1., top_s1, top_s2, top_th),
                (-1., bot_s1, bot_s2, bot_th)]:
            z = sign * 0.5 * t
            sx  = Nx / t + (12. * z / t**3) * Mx
            sy  = Ny / t + (12. * z / t**3) * My
            txy = Nxy / t + (12. * z / t**3) * Mxy
            s1_arr[e], s2_arr[e], th_arr[e] = principal_stresses(sx, sy, txy)
    return top_s1, top_s2, top_th, bot_s1, bot_s2, bot_th

def _nodal_average_scalar(node_tags, tri6_conn, node_id_map, elem_values):
    """Average element-centroid scalar values to nodes (corner nodes only)."""
    vals   = np.zeros(len(node_tags))
    counts = np.zeros(len(node_tags))
    for e in range(len(tri6_conn) // 6):
        for tag in tri6_conn[6*e:6*e + 3]:   # corner nodes only
            idx = node_id_map[tag]
            vals[idx]   += elem_values[e]
            counts[idx] += 1.0
    counts[counts == 0] = 1.0
    return vals / counts

def _nodal_average_response(node_tags, tri6_conn, node_id_map, responses, field, comp):
    elem_vals = np.array([responses[e][field][comp] for e in range(len(responses))])
    return _nodal_average_scalar(node_tags, tri6_conn, node_id_map, elem_vals)

def _triangulation(node_tags, coords_dict, tri6_conn, node_id_map):
    x = np.array([coords_dict[t][0] for t in node_tags])
    y = np.array([coords_dict[t][1] for t in node_tags])
    tri = [[node_id_map[tri6_conn[6*e]], node_id_map[tri6_conn[6*e+1]], node_id_map[tri6_conn[6*e+2]]]
           for e in range(len(tri6_conn) // 6)]
    return x, y, Triangulation(x, y, tri)

def plot_contour_displacement(node_tags, coords_dict, tri6_conn, node_id_map, U, component="w"):
    comp_idx = {"u": 0, "v": 1, "w": 2, "rx": 3, "ry": 4}[component]
    values   = np.array([U[node_id_map[t] * 5 + comp_idx] for t in node_tags])
    x, y, triang = _triangulation(node_tags, coords_dict, tri6_conn, node_id_map)
    fig, ax = plt.subplots(figsize=(10, 7))
    tcf = ax.tricontourf(triang, values, levels=30, cmap="viridis")
    plt.colorbar(tcf, ax=ax, label=f"{component} [mm]")
    ax.set_aspect("equal"); ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"Displacement — {component}")
    plt.tight_layout(); plt.show()

def plot_contour_resultant(node_tags, coords_dict, tri6_conn, node_id_map,
                           responses, field="M", comp=0, label=None):
    """field: 'Nm'|'M'|'Q'|'eps_m'|'kappa'|'gamma',  comp: 0,1,2"""
    nodal = _nodal_average_response(node_tags, tri6_conn, node_id_map, responses, field, comp)
    x, y, triang = _triangulation(node_tags, coords_dict, tri6_conn, node_id_map)
    title = label or f"{field}[{comp}]"
    fig, ax = plt.subplots(figsize=(10, 7))
    tcf = ax.tricontourf(triang, nodal, levels=30, cmap="plasma")
    plt.colorbar(tcf, ax=ax, label=title)
    ax.set_aspect("equal"); ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"Contour — {title}")
    plt.tight_layout(); plt.show()

def plot_contour_surface_stress(node_tags, coords_dict, tri6_conn, node_id_map,
                                responses, t, surface="top", quantity="sigma1"):
    """Plot nodal-averaged principal stress on top or bottom surface."""
    top_s1, top_s2, _, bot_s1, bot_s2, _ = surface_stresses(responses, t)
    data_map = {"top": {"sigma1": top_s1, "sigma2": top_s2},
                "bot": {"sigma1": bot_s1, "sigma2": bot_s2}}
    elem_vals = data_map[surface][quantity]
    nodal = _nodal_average_scalar(node_tags, tri6_conn, node_id_map, elem_vals)
    x, y, triang = _triangulation(node_tags, coords_dict, tri6_conn, node_id_map)
    title = f"{surface} surface σ₁" if quantity == "sigma1" else f"{surface} surface σ₂"
    fig, ax = plt.subplots(figsize=(10, 7))
    tcf = ax.tricontourf(triang, nodal, levels=30, cmap="RdBu_r")
    plt.colorbar(tcf, ax=ax, label="[MPa]")
    ax.set_aspect("equal"); ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(title)
    plt.tight_layout(); plt.show()

def plot_deformed_shape(node_tags, coords_dict, tri6_conn, node_id_map, U, scale=1.0):
    x = np.array([coords_dict[t][0] for t in node_tags])
    y = np.array([coords_dict[t][1] for t in node_tags])
    w = np.array([U[node_id_map[t] * 5 + 2] for t in node_tags])
    tri = [[node_id_map[tri6_conn[6*e]], node_id_map[tri6_conn[6*e+1]], node_id_map[tri6_conn[6*e+2]]]
           for e in range(len(tri6_conn) // 6)]
    triang = Triangulation(x, y + scale * w, tri)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.triplot(triang, color="k", lw=0.5)
    ax.set_aspect("equal"); ax.set_xlabel("X"); ax.set_ylabel("Y + scale·w")
    ax.set_title(f"Deformed shape  (scale={scale})")
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout(); plt.show()

def export_xdmf(node_tags, coords_dict, tri6_conn, node_id_map, U, responses, t,
                filename_base="results"):
    """Export displacement, rotations, and surface stresses to XDMF/H5 for ParaView."""
    points    = np.array([coords_dict[tag] for tag in node_tags])
    triangles = np.array([[node_id_map[tri6_conn[6*e + k]] for k in range(3)]
                          for e in range(len(tri6_conn) // 6)])

    disp = np.zeros((len(node_tags), 3))
    rot  = np.zeros((len(node_tags), 2))
    for i, tag in enumerate(node_tags):
        base = node_id_map[tag] * 5
        disp[i] = U[base:base + 3]
        rot[i]  = U[base + 3:base + 5]

    top_s1, top_s2, _, bot_s1, bot_s2, _ = surface_stresses(responses, t)

    Mx_nodal    = _nodal_average_response(node_tags, tri6_conn, node_id_map, responses, "M", 0)
    top_s1_n    = _nodal_average_scalar(node_tags, tri6_conn, node_id_map, top_s1)
    top_s2_n    = _nodal_average_scalar(node_tags, tri6_conn, node_id_map, top_s2)
    bot_s1_n    = _nodal_average_scalar(node_tags, tri6_conn, node_id_map, bot_s1)
    bot_s2_n    = _nodal_average_scalar(node_tags, tri6_conn, node_id_map, bot_s2)

    mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
        point_data={
            "displacement":  disp,
            "rotation":      rot,
            "Mx":            Mx_nodal,
            "top_sigma1":    top_s1_n,
            "top_sigma2":    top_s2_n,
            "bot_sigma1":    bot_s1_n,
            "bot_sigma2":    bot_s2_n,
        },
    )
    meshio.write(f"{filename_base}.xdmf", mesh)
    print(f"Exported {filename_base}.xdmf + .h5 for ParaView.")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    gmsh.initialize()
    try:
        create_geometry(
            plate_w=PLATE_W, plate_h=PLATE_H,
            cutout_type=CUTOUT_TYPE,
            cx=CX, cy=CY, w=CUTOUT_W, h=CUTOUT_H, r=CUTOUT_R,
        )
        debug_print_gmsh_groups()

        if VERIFY_GEOMETRY_ONLY:
            print("Geometry OK — stopping before FE solve (VERIFY_GEOMETRY_ONLY=True).")
        else:
            # --- Mesh ---
            node_tags, coords_dict, tri6_conn = create_tri6_mesh(
                mesh_size=MESH_SIZE, msh_file="plate_mitc6.msh", plot=True
            )

            # --- Stiffness ---
            K, node_id_map = assemble_stiffness(
                node_tags, coords_dict, tri6_conn, E, NU, THICKNESS
            )
            # Small numerical regularisation (removes any remaining zero-energy modes)
            K = K + 1e-12 * identity(K.shape[0], format="csr")

            # --- Load: uniform lateral pressure ---
            F = assemble_pressure_load(
                node_tags, coords_dict, tri6_conn, node_id_map, PRESSURE
            )

            # --- Boundary conditions ---
            # Left edge: fully clamped (all 5 DOFs fixed)
            left_nodes  = get_nodes_on_physical_group(1, "LEFT")
            fixed       = fixed_dofs_for_nodes(node_id_map, left_nodes, dof_indices=(0,1,2,3,4))

            # Suppress in-plane rigid-body translation on remaining edges
            # (one u-DOF on right, one v-DOF on bottom)
            right_nodes  = get_nodes_on_physical_group(1, "RIGHT")
            bottom_nodes = get_nodes_on_physical_group(1, "BOTTOM")
            if right_nodes:
                fixed.append(node_id_map[right_nodes[0]] * 5 + 0)
            if bottom_nodes:
                fixed.append(node_id_map[bottom_nodes[0]] * 5 + 1)
            fixed = sorted(set(fixed))

            K_ff, F_f, free_dofs, fixed_dofs, u_presc = apply_bcs(K, F, fixed)
            print(f"System: {K_ff.shape[0]} free DOFs, {len(fixed_dofs)} fixed DOFs")
            print(f"|F|  = {np.linalg.norm(F_f):.4e}")

            # --- Solve ---
            U = solve(K_ff, F_f, free_dofs, fixed_dofs, u_presc, K.shape[0])
            if not np.all(np.isfinite(U)):
                raise RuntimeError("Solution contains NaN/Inf — check BCs or mesh.")
            print(f"Max |w|   = {np.max(np.abs(U[2::5])):.4e} mm")
            print(f"Max |u|   = {np.max(np.abs(U[0::5])):.4e} mm")
            print(f"Max |v|   = {np.max(np.abs(U[1::5])):.4e} mm")

            # --- Stress recovery ---
            responses = recover_all_elements(
                node_tags, coords_dict, tri6_conn, node_id_map, U, E, NU, THICKNESS
            )
            r0 = responses[0]
            print(f"Element 0 — Nm={r0['Nm']}  M={r0['M']}  Q={r0['Q']}")

            # --- Export ---
            export_xdmf(
                node_tags, coords_dict, tri6_conn, node_id_map, U, responses,
                THICKNESS, filename_base="plate_results"
            )

            # --- Plots ---
            plot_contour_displacement(node_tags, coords_dict, tri6_conn, node_id_map, U, "w")
            plot_contour_resultant(node_tags, coords_dict, tri6_conn, node_id_map,
                                   responses, field="M", comp=0, label="Mx [N·mm/mm]")
            plot_contour_surface_stress(node_tags, coords_dict, tri6_conn, node_id_map,
                                        responses, THICKNESS, surface="top", quantity="sigma1")
            plot_deformed_shape(node_tags, coords_dict, tri6_conn, node_id_map, U, scale=50.0)
    finally:
        gmsh.finalize()
