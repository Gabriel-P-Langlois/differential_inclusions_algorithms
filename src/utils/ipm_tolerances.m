function tol = ipm_tolerances()
%IPM_TOLERANCES  Tolerances of the interior-point solvers (MOSEK, Gurobi,
%   HiGHS) used in both PDE-constrained optimal control drivers.
%
%   Each value is the solver's floor.  MOSEK rejects 1e-12 (rcode 100006),
%   HiGHS errors below 1e-10, and Gurobi's feasibility and optimality
%   tolerances stop at 1e-9.
%
%   OUTPUT
%       tol.mosek_qo    -   MOSEK MSK_DPAR_INTPNT_QO_TOL_PFEAS, _DFEAS and
%                           _REL_GAP.
%       tol.gurobi_bar  -   Gurobi BarConvTol.
%       tol.gurobi_fo   -   Gurobi FeasibilityTol and OptimalityTol.
%       tol.highs       -   HiGHS ipm_optimality_tolerance and the primal
%                           and dual feasibility tolerances.

    tol = struct('mosek_qo', 1e-10, 'gurobi_bar', 1e-14, ...
                 'gurobi_fo', 1e-9, 'highs', 1e-10);
end
