%   TEST_PHASE1     This script verifies that di_phase1.m correctly finds
%                   a feasible point for a feasible linear inequality
%                   system and correctly certifies infeasibility for an
%                   infeasible one.
%
%                   The system in question is given by A*x <= b.
%
% -------------------------------------------------------------------------
%   FUNCTIONS THAT ARE TESTED
%       [x, flag] = di_phase1(A, b, tol, [], @epgd_lsqnonneg)
%           -   Differential inclusions Phase I with exact projected
%               gradient descent.
%
%       [x, flag] = di_phase1(A, b, tol, [], @apgd_lsqnonneg)
%           -   Differential inclusions Phase I with accelerated projected
%               gradient descent.
%
% -------------------------------------------------------------------------


%% Preliminaries
rng(1);

m = 500;
n = 1000;

x0 = randn(n, 1);
A  = randn(m, n);
s  = abs(randn(m, 1)) + 1;
b  = A * x0 + s;

tol = min(1e-08, 100*eps*max(norm(A, 'fro'), 1)*max(norm(b), 1));


%% Test 1: Feasible system.
fprintf("Phase I Test 1: Feasible system.\n\n")

fprintf("Running di_phase1 with epgd_lsqnonneg...\n")
[x_epgd, flag_epgd] = di_phase1(A, b, tol, [], @epgd_lsqnonneg);
fprintf("Done.\n\n")

fprintf("Running di_phase1 with apgd_lsqnonneg...\n")
[x_apgd, flag_apgd] = di_phase1(A, b, tol, [], @apgd_lsqnonneg);
fprintf("Done.\n\n")

assert(flag_epgd == 1, 'di_phase1 (epgd): expected flag = 1 for feasible system.')
assert(flag_apgd == 1, 'di_phase1 (apgd): expected flag = 1 for feasible system.')
assert(all(A*x_epgd <= b + tol), 'di_phase1 (epgd): returned point is not feasible.')
assert(all(A*x_apgd <= b + tol), 'di_phase1 (apgd): returned point is not feasible.')

fprintf("Test 1 passed: both solvers returned a feasible point.\n\n")


%% Test 2: Infeasible system.
% The system requires x1 >= 1 and x1 <= 0, which is infeasible.
fprintf("Phase I Test 2: Infeasible system.\n\n")

A_inf = [1; -1];
b_inf = [0; -1];

fprintf("Running di_phase1 (epgd) on infeasible system...\n")
[x_inf_epgd, flag_inf_epgd] = di_phase1(A_inf, b_inf, 1e-08, [], ...
    @epgd_lsqnonneg);
fprintf("Done.\n\n")

fprintf("Running di_phase1 (apgd) on infeasible system...\n")
[x_inf_apgd, flag_inf_apgd] = di_phase1(A_inf, b_inf, 1e-08, [], ...
    @apgd_lsqnonneg);
fprintf("Done.\n\n")

assert(flag_inf_epgd == 0, 'di_phase1 (epgd): expected flag = 0 for infeasible system.')
assert(flag_inf_apgd == 0, 'di_phase1 (apgd): expected flag = 0 for infeasible system.')
assert(isempty(x_inf_epgd), 'di_phase1 (epgd): expected empty x for infeasible system.')
assert(isempty(x_inf_apgd), 'di_phase1 (apgd): expected empty x for infeasible system.')

fprintf("Test 2 passed: both solvers correctly certified infeasibility.\n\n")
