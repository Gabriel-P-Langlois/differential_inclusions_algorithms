# Continuous-time analysis and algorithms for linear and convex quadratic programs via exactly solvable differential inclusions

Gabriel P. Langlois, Akwum Onwunta, and Jérôme Darbon

This repository holds the MATLAB code of the paper's numerical experiments (Section 6): the solvers of Algorithms 1 and 2, the four experiment drivers, and the saved runs from which the paper's tables and figures were made.

## Requirements

- MATLAB. We used R2025b on an Apple M3 with 16 GB of RAM. No toolbox is required.
- Optional, for the comparisons of Section 6.2: [MOSEK](https://www.mosek.com) 11.2, [Gurobi](https://www.gurobi.com) 13.0.2, and [HiGHS](https://highs.dev) 1.14.0 through the MATLAB interface [HiGHSMEX](https://github.com/savyasachi/HiGHSMEX) 1.4.0. MOSEK and Gurobi need a license, free for academic use. A driver skips any solver it cannot find.

The two drivers of Section 6.2 look for the optional solvers here; edit the `%% Paths` block at the top of each driver to match your installation:

```matlab
p_mosek  = fullfile(getenv('HOME'), 'mosek', '11.2', 'toolbox', 'r2022b');
p_gurobi = '/Library/gurobi1302/macos_universal2/matlab';
p_highs  = fullfile(getenv('HOME'), 'HiGHSMEX');
```

## Repository layout

```
src/
  main/
    di_pos_w.m        Algorithm 1 for the weighted limiter QP (Section 6.1)
    di_qp.m           Algorithm 2 (minimal selection descent), SPD Q
    di_lp.m           Algorithm 1 with t = 0, i.e. linear programs
    di_phase1.m       feasible starting point, called by di_qp and di_lp
  utils/
    ipm_tolerances.m  tolerances of MOSEK, Gurobi, and HiGHS
    nnls/             NNLS subproblem solvers: box_lsqnonneg (used in
                      Section 6.2), apgd_lsqnonneg (default of di_qp, di_lp)
    limiter/          limiter QP: DR algorithm, starting point, reference
    dg/               discontinuous Galerkin solver of Section 6.1.2
    pde_oc/           optimal control problems of Section 6.2
results/
  benchmark_*.m       experiment drivers, one per subsection of Section 6
  make_*.m            write the tables and figures from the saved runs
  pos_synthetic_runs/ pos_PDE_runs/ pde_oc_runs/   saved runs (.mat, .log)
  tables/             LaTeX table rows, as in the paper
  figures/            PDF figures, as in the paper
```

## Running the experiments

Each driver adds the paths it needs. Run it from MATLAB in `results/`:

| Section | Experiment | Command | Run time on our machine |
|---|---|---|---|
| 6.1.1 | Limiter QP on synthetic data | `benchmark_pos_synthetic` | 33 s |
| 6.1.2 | Limiter in a DG scheme | `benchmark_pos_PDE` | about 15 h |
| 6.2.1 | 3D Poisson optimal control | `benchmark_pde_oc_3D` | 1 h 16 min |
| 6.2.2 | 2D heat optimal control | `benchmark_pde_oc_2D` | 4 h 13 min |

The run times come from the time stamps of the saved runs and include all solvers.

Each driver saves its results in its folder under `results/` after every mesh or grid. Then write the tables and figures:

```matlab
make_pos_tables;           make_pos_figures             % Section 6.1
make_pde_oc_tables('3D');  make_pde_oc_figures('3D')    % Section 6.2.1
make_pde_oc_tables('2D');  make_pde_oc_figures('2D')    % Section 6.2.2
```

These scripts read the newest run in each folder and write to `results/tables/` and `results/figures/`. To rebuild the paper's tables and figures without running the experiments, call them on the saved runs that ship with the repository.

| Section | Saved run | Tables | Figures |
|---|---|---|---|
| 6.1.1 | `pos_synthetic_runs/pos_synthetic_2026-09-16_094043` | `pos_synth_time`, `pos_synth_acc` | `pos_synthetic_time` |
| 6.1.2 | `pos_PDE_runs/k*_n*_{dipos,dr}` | `pos_dg` | — |
| 6.2.1 | `pde_oc_runs/pde_oc_3D_2026-09-17_105235` | `pde_oc_3D_pair{1,2}_{time,acc,paper}` | `pde_oc_3D_pair{1,2}_time` |
| 6.2.2 | `pde_oc_runs/pde_oc_2D_2026-09-17_230708` | `pde_oc_2D_pair{1,2}_{time,acc,paper}` | `pde_oc_2D_pair{1,2}_time` |

A new run of `benchmark_pos_PDE` overwrites the files in `pos_PDE_runs/`; the other drivers add a dated run next to the saved one.

## Data not stated in the source papers

The experiments follow Liu, Hu, Taitano, and Zhang (Section 6.1) and Pearson and Gondzio (Section 6.2). Where these papers leave a datum unstated, we chose it. The header of each driver lists every such choice and its source; the main ones are:

- Section 6.1.1: the grid points are the lower-left cell corners, and δ = 0.05.
- Section 6.1.2: h = Δx in both the penalty and the norm of the DR stopping rule; Δτ = 19/119 for k = 3 and n = 64, since their Δτ = 0.16 does not split the time interval [1, 20] into whole steps; ε = 0 in the Zhang–Shu limiter.
- Section 6.2.1: assumptions A1–A3 (boundary condition, grid, desired-state term).
- Section 6.2.2: assumptions A1–A5 (domain, boundary and initial conditions, desired state with amplitude 0.5, desired-state term, grid), and Δτ = 0.02.

## References

- C. Liu, J. Hu, W. T. Taitano, and X. Zhang, An optimization-based positivity-preserving limiter in semi-implicit discontinuous Galerkin schemes solving Fokker–Planck equations, Comput. Math. Appl. 192 (2025), 54–71.
- J. W. Pearson and J. Gondzio, Fast interior point solution of quadratic programming problems arising from PDE-constrained optimization, Numer. Math. 137 (2017), 959–999.
- IFISS 3.6, whose Q1 grid and desired-state conventions the drivers of Section 6.2 follow.

## Citation

A BibTeX entry will be added once the paper is available.

## Disclaimer

We wrote this code with the assistance of Claude (Anthropic), an AI model. It may contain minor deviations from the paper.

## License

Released under the MIT License; see `LICENSE`.
