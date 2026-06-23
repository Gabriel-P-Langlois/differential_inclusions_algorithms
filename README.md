# Numerical Code — Gradient Inclusions for Linear and Convex Quadratic Programs

MATLAB code accompanying the paper "Gradient Inclusions for Linear and Convex Quadratic Programs" by G. P. Langlois and J. Darbon.

## Directory structure

```
numerics/
├── src/
│   ├── main/          # DI solvers
│   └── utils/
│       ├── nnls/      # NNLS sub-solvers
│       └── prox/      # Proximal operators (auxiliary)
├── test/
│   ├── lp/            # LP correctness and speed tests
│   ├── nnls/          # NNLS solver correctness tests
│   ├── qp/            # QP correctness tests and PDE-constrained benchmarks
│   └── rlp/           # Regularized LP tests
└── results/           # Saved output (figures, .mat files) from benchmark scripts
```

## Main solvers (`src/main/`)

| File | Description |
|---|---|
| `di_lp.m` | DI solver for LP; accepts any NNLS solver via function handle |
| `di_rlp.m` | DI solver for L2-regularized LP: min (t/2)\|\|x\|\|² + c'x s.t. Ax ≤ b, t > 0 (Algorithm 1, finite-time) |
| `di_qp.m` | DI solver for convex QP with SPD Q (Algorithm 2); Q may be a matrix or a function handle |
| `di_qp_eq.m` | DI solver for min (1/2)x'Qx + c'x s.t. Ax ≤ b, Ex = f; equality constraints handled via null-space projection |
| `di_phase1.m` | Feasibility phase; solves an augmented LP via `di_lp` |

## NNLS sub-solvers (`src/utils/nnls/`)

These implement `min_{q >= 0} (1/2)||Aq - b||²` and are passed as function handles to the main solvers.

| File | Method |
|---|---|
| `epgd_lsqnonneg.m` | Exact projected gradient descent (exact line search); warm-start capable |
| `apgd_lsqnonneg.m` | FISTA with gradient restart; warm-start capable |
| `pgd_lsqnonneg.m` | Projected gradient descent (fixed step) |
| `hinges_lsqnonneg.m` | Method of hinges (Mayer); LSQR postprocess for large sparse problems |
| `pcg_lsqnonneg.m` | Gradient projection CG (Moré–Toraldo 1991) |
| `lbfgs_lsqnonneg.m` | L-BFGS (not recommended for poorly-scaled sparse data) |

## Tests and benchmarks (`test/`)

### NNLS (`test/nnls/`)
- `test_nnls.m` — correctness of all NNLS solvers (`epgd`, `apgd`, `pgd`, `hinges`, `lbfgs`, `pcg`) on dense small data
- `test_nnls_dense.m` — dense data regression for `epgd`, `apgd`, and `lbfgs`
- `test_nnls_sparse.m` — sparse data regression for `epgd`, `apgd`, and `lbfgs`

### LP (`test/lp/`)
- `test_lp.m` — correctness of `di_lp` with all six NNLS solvers
- `test_lp_speed.m` — timing comparison (`apgd`, `lbfgs`, `pcg`)
- `test_phase1.m` — correctness of `di_phase1` (feasible and infeasible cases)

### QP (`test/qp/`)
- `test_qp.m` — correctness of `di_qp` vs `quadprog` (m=100, n=200, SPD Q)
- `test_qp_fun_handle.m` — verifies that a function-handle Q agrees with the explicit matrix form
- `test_qp_eq.m` — correctness of `di_qp_eq`; checks all four KKT conditions

**PDE-constrained benchmarks** (Poisson optimal control, Pearson & Gondzio 2017):

| File | Problem | Desired state | Constraints | Mass matrix |
|---|---|---|---|---|
| `benchmark_pde_oc_2D_I.m` | 2D, single mesh (n=32) | Gaussian bump | control only | lumped (h²I) |
| `benchmark_pde_oc_2D_II.m` | 2D, single mesh (n=32) | sin(πx₁)sin(πx₂) | state only | lumped (h²I) |
| `benchmark_pde_oc_2D_III.m` | 2D, mesh sweep n∈{16,32,48,64} | sin(πx₁)sin(πx₂) | state only | lumped (h²I) |
| `benchmark_pde_oc_3D_I.m` | 3D, mesh sweep | Gaussian bump | control only | consistent Q1 |

All benchmarks compare `di_qp_eq` (direct Cholesky) against MOSEK. The 2D benchmarks also include `quadprog`. `benchmark_pde_oc_3D_I.m` additionally compares `di_qp` applied to the null-space-reduced u-only QP (equality constraint K*y = M_c*u eliminated by substituting y = K⁻¹M_c*u). Reference: Pearson & Gondzio (2017), *Numer. Math.* 137:959–999.

The stiffness matrix K is identical across all benchmarks and matches the IFISS-assembled matrix from the paper. `benchmark_pde_oc_3D_I.m` uses the consistent Q1 mass matrix M_c = M1D ⊗ M1D ⊗ M1D, matching the paper's discretization; the 2D benchmarks use the row-sum lumped mass M = hᵈI. The two discretizations agree to O(h²).

### RLP (`test/rlp/`)
- `test_rlp.m` — correctness of `di_rlp` vs `quadprog`; includes KKT checks
