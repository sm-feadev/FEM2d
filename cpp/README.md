# MITC6 Shell FEM — C++ solver

MITC6 Reissner–Mindlin shell element solver for plates with an **arbitrary number
of cutouts**.  Units throughout: **mm · N · MPa**.

## DOF layout

Each node carries 5 DOFs:

| Index | Symbol | Meaning |
|-------|--------|---------|
| 0 | u  | in-plane translation, x |
| 1 | v  | in-plane translation, y |
| 2 | w  | transverse deflection   |
| 3 | θx | rotation about x        |
| 4 | θy | rotation about y        |

## Files

| File | Purpose |
|------|---------|
| `mitcs6.h`   | All types, structs, and function declarations |
| `mitcs6.cpp` | Full implementation |
| `main.cpp`   | Driver — configure geometry, BCs, load, solve |
| `CMakeLists.txt` | CMake build |

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| [Eigen](https://eigen.tuxfamily.org) | ≥ 3.4 | Dense + sparse linear algebra |
| [gmsh](https://gmsh.info) | ≥ 4.0 | Geometry (OCC) + TRI6 meshing |

### Install on Ubuntu/Debian

```bash
sudo apt install libeigen3-dev libgmsh-dev
```

### Install on macOS (Homebrew)

```bash
brew install eigen gmsh
```

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Adjust scenario in `main.cpp` (`SCENARIO = 1 / 2 / 3`) before building.

## Run

```bash
./mitcs6
```

Writes `plate_results.vtk` — open in [ParaView](https://www.paraview.org).

## Cutout types

Defined by `CutoutType` enum and `Cutout` struct:

```cpp
struct Cutout {
    CutoutType type;   // Circle | Ellipse | Rectangle | RoundedRectangle
    double cx, cy;     // centre [mm]
    double w, h;       // width / height (or unused for circle)
    double r;          // corner radius (RoundedRectangle) or radius (Circle)
};
```

### Arbitrary number of cutouts

```cpp
cfg.cutouts.push_back({CutoutType::Circle,          80, 100, 0, 0, 25});
cfg.cutouts.push_back({CutoutType::Ellipse,         180, 100, 80, 40, 0});
cfg.cutouts.push_back({CutoutType::RoundedRectangle,260, 100, 60, 50, 8});
```

Each cutout gets its own physical boundary group `HOLE_0`, `HOLE_1`, … (plus
a combined `HOLE` group) so you can apply different BCs per hole edge.

## Boundary condition masks

```cpp
add_fixed_dofs(mesh, nodes, DOF_ALL,  fixed);  // clamped
add_fixed_dofs(mesh, nodes, DOF_U|DOF_V|DOF_W, fixed);  // pinned
add_fixed_dofs(mesh, nodes, DOF_W,    fixed);  // simply supported
add_fixed_dofs(mesh, nodes, DOF_U,    fixed);  // roller in x
add_fixed_dofs(mesh, nodes, DOF_U|DOF_TY, fixed); // symmetry about x=const
```

| Constant | Value | DOF |
|----------|-------|-----|
| `DOF_U`  | bit 0 | u   |
| `DOF_V`  | bit 1 | v   |
| `DOF_W`  | bit 2 | w   |
| `DOF_TX` | bit 3 | θx  |
| `DOF_TY` | bit 4 | θy  |
| `DOF_ALL`| 0x1F  | all |

## VTK output fields

| Field | Type | Description |
|-------|------|-------------|
| `displacement` | vector | (u, v, w) [mm] |
| `rotation`     | vector | (θx, θy, 0) [rad] |
| `w_deflection` | scalar | transverse deflection [mm] |
| `top_sigma1`   | scalar | top fibre σ₁ [MPa] |
| `top_sigma2`   | scalar | top fibre σ₂ [MPa] |
| `bot_sigma1`   | scalar | bottom fibre σ₁ [MPa] |
| `bot_sigma2`   | scalar | bottom fibre σ₂ [MPa] |
| `von_mises_top`| scalar | von Mises stress, top [MPa] |
| `von_mises_bot`| scalar | von Mises stress, bottom [MPa] |
| `Mx`, `My`     | scalar | bending moment resultants [N·mm/mm] |
| `Nx`, `Ny`     | scalar | membrane stress resultants [N/mm] |

## Validation benchmarks

| Case | Reference | Expected |
|------|-----------|---------|
| SS square plate, uniform pressure | Timoshenko §44 | w_max = 0.00406·pL⁴/(Et³) |
| Clamped square plate | Timoshenko §44 | w_max = 0.00126·pL⁴/(Et³) |
| Circular plate, clamped, point load | Timoshenko §15 | w_max = PL²/(16πD) |
| Mesh convergence rate | — | displacements O(h³), stresses O(h²) |
