%   TEST_QP_EQ  Verifies that di_qp_eq.m solves equality- and inequality-
%               constrained convex QPs correctly, and reports timings.
%
%               Problem:
%                   min  (1/2)*x'*Q*x + c'*x
%                   s.t. A*x <= b,  E*x = f,
%               where Q is symmetric positive definite.
%
%               Checks four KKT conditions at the computed solution:
%                   1. Primal gap            |fval_quadprog - fval_di|
%                   2. Equality feasibility  norm(E*x - f, inf)
%                   3. Stationarity          norm(Q*x + c + A'*p + E'*mu, inf)
%                   4. Complementary slack   norm(p .* (b - A*x), inf)
%
% -------------------------------------------------------------------------

% Ensure MATLAB's quadprog takes priority over MOSEK's shadowing version.
addpath(fullfile(matlabroot, 'toolbox', 'optim', 'optim'));


%% Problem data
rng(1);
m  = 200;
n  = 100;
me = 20;

% Strictly feasible interior point.
x0 = randn(n, 1);
A  = randn(m, n);
s  = abs(randn(m, 1)) + 1;
b  = A*x0 + s;

% Equality constraints satisfied at x0.
E  = randn(me, n);
f  = E*x0;

% Random SPD matrix and cost vector.
B = randn(n, n);
Q = B'*B + eye(n);
c = randn(n, 1);

tol = min(1e-8, 100*eps*max(norm(A,'fro'),1)*max(norm(b),1));


%% Reference solution via quadprog

fprintf('Solving with quadprog...\n')
opts = optimoptions('quadprog', 'Display', 'off', ...
    'OptimalityTolerance', tol, 'Algorithm', 'interior-point-convex');
tic
[x_qp, fval_qp] = quadprog(Q, c, A, b, E, f, [], [], [], opts);
time_qp = toc;
fprintf('Done.  time = %.4f s\n\n', time_qp)


%% Test 1: di_qp_eq with default PE_fun (Cholesky)

fprintf('Test 1: di_qp_eq with default PE_fun.\n')
tic
[x_di, p_di, mu_di, fval_di, flag_di] = di_qp_eq(A, b, c, Q, E, f, x0, tol);
time_di1 = toc;
fprintf('Done.  flag = %d,  time = %.4f s\n\n', flag_di, time_di1)

assert(flag_di == 1, 'di_qp_eq did not converge.')

primal_gap   = abs(fval_qp - fval_di);
eq_feas      = norm(E*x_di - f, inf);
stationarity = norm(Q*x_di + c + A'*p_di + E'*mu_di, inf);
comp_slack   = norm(p_di .* (b - A*x_di), inf);

fprintf('  primal gap:            %e\n', primal_gap)
fprintf('  equality feasibility:  %e\n', eq_feas)
fprintf('  stationarity:          %e\n', stationarity)
fprintf('  complementary slack:   %e\n\n', comp_slack)

assert(primal_gap   < 1e4*tol, 'Primal gap too large.')
assert(eq_feas      < 1e4*tol, 'Equality constraint violated.')
assert(stationarity < 1e4*tol, 'Stationarity condition not satisfied.')
assert(comp_slack   < 1e4*tol, 'Complementary slackness not satisfied.')


%% Test 2: di_qp_eq with user-supplied PE_fun (same Cholesky, explicit)

fprintf('Test 2: di_qp_eq with user-supplied PE_fun.\n')
EEt = E * E';
if issparse(E)
    [L, ~, perm] = chol(EEt, 'lower', 'vector');
    PE_user = @(v) pe_apply_test_sparse(E, L, perm, me, v);
else
    [R, ~] = chol(EEt);
    PE_user = @(v) pe_apply_test_dense(E, R, v);
end

tic
[x_di2, p_di2, mu_di2, fval_di2, flag_di2] = ...
    di_qp_eq(A, b, c, Q, E, f, x0, tol, [], PE_user);
time_di2 = toc;
fprintf('Done.  flag = %d,  time = %.4f s\n\n', flag_di2, time_di2)

assert(flag_di2 == 1, 'di_qp_eq (user PE_fun) did not converge.')
assert(abs(fval_di2 - fval_di) < 1e4*tol, ...
    'User PE_fun result differs from default.')
fprintf('  fval difference (default vs user PE_fun): %e\n\n', ...
    abs(fval_di2 - fval_di))


%% Test 3: Q supplied as a function handle

fprintf('Test 3: di_qp_eq with Q as a function handle.\n')
Qfun = @(v) Q*v;
tic
[~, ~, ~, fval_di3, flag_di3] = di_qp_eq(A, b, c, Qfun, E, f, x0, tol);
time_di3 = toc;
fprintf('Done.  flag = %d,  time = %.4f s\n\n', flag_di3, time_di3)

assert(flag_di3 == 1, 'di_qp_eq (Q handle) did not converge.')
assert(abs(fval_di3 - fval_di) < 1e4*tol, ...
    'Q function-handle result differs from matrix form.')
fprintf('  fval difference (matrix Q vs Q handle): %e\n\n', ...
    abs(fval_di3 - fval_di))


%% Summary

fprintf('=== Summary ===\n')
fprintf('  quadprog fval:          %.10f   time = %.4f s\n', fval_qp,  time_qp)
fprintf('  di_qp_eq (default):     %.10f   time = %.4f s\n', fval_di,  time_di1)
fprintf('  di_qp_eq (user PE_fun): %.10f   time = %.4f s\n', fval_di2, time_di2)
fprintf('  di_qp_eq (Q handle):    %.10f   time = %.4f s\n', fval_di3, time_di3)
fprintf('  primal gap (Test 1):    %e\n', primal_gap)
fprintf('All tests passed.\n')


%% Local helpers for Test 2

function out = pe_apply_test_dense(E, R, v)
    out = v - E' * (R \ (R' \ (E * v)));
end

function out = pe_apply_test_sparse(E, L, perm, me, v)
    Ev        = E * v;
    tmp       = zeros(me, 1);
    tmp(perm) = L' \ (L \ Ev(perm));
    out       = v - E' * tmp;
end
