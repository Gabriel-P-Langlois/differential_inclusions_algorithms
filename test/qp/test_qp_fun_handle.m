%   TEST_QP_FUN_HANDLE  Verifies that di_qp accepts a function handle for
%                       Q and produces the same result as passing the
%                       explicit matrix.
%
%   The quadratic program is
%       min_{x} (1/2)*x'*(B'*B + mu*I)*x + c'*x   subject to A*x <= b,
%   where Q = B'*B + mu*I is symmetric positive definite.  Rather than
%   forming Q explicitly, the function handle @(v) B'*(B*v) + mu*v
%   computes the matrix-vector product without materializing the n x n
%   Gram matrix B'*B.  Both calls to di_qp should yield the same fval
%   within the prescribed tolerance.
%
% -------------------------------------------------------------------------

%% Preliminaries
rng(1);

% Ensure MATLAB's quadprog takes priority over MOSEK's shadowing version.
addpath(fullfile(matlabroot, 'toolbox', 'optim', 'optim'));

% Problem dimensions
p  = 30;
m  = 50;
n  = 100;
mu = 1;

% Factors defining Q = B'*B + mu*I
B = randn(p, n);

% Function handle: avoids forming the n x n matrix B'*B.
Q_fun = @(v) B'*(B*v) + mu*v;

% Explicit matrix (for reference run and quadprog baseline).
Q_mat = B'*B + mu*eye(n);

% Feasibility data: x0 is a strict interior point.
x0 = randn(n, 1);
A  = randn(m, n);
s  = abs(randn(m, 1)) + 1;
b  = A*x0 + s;
c  = randn(n, 1);

tol = min(1e-08, 100*eps*max(norm(A,'fro'), 1)*max(norm(b), 1));


%% Baseline: MATLAB quadprog
fprintf('Running quadprog (baseline)...\n')
opts = optimoptions('quadprog', 'Display', 'off', ...
    'OptimalityTolerance', tol, 'Algorithm', 'interior-point-convex');
tic
[~, fval_qp] = quadprog(Q_mat, c, A, b, [], [], [], [], [], opts);
time_qp = toc;
fprintf('Done.\n\n')


%% di_qp with explicit Q matrix
fprintf('Running di_qp with explicit Q matrix...\n')
tic
[~, ~, fval_mat, flag_mat] = di_qp(A, b, c, Q_mat, x0, tol, [], @apgd_lsqnonneg);
time_mat = toc;
fprintf('Done.\n\n')


%% di_qp with function handle
fprintf('Running di_qp with function handle @(v) B''*(B*v) + mu*v...\n')
tic
[~, ~, fval_fun, flag_fun] = di_qp(A, b, c, Q_fun, x0, tol, [], @apgd_lsqnonneg);
time_fun = toc;
fprintf('Done.\n\n')


%% Results
fprintf('Flags:  matrix = %d,  function handle = %d\n\n', flag_mat, flag_fun)

disp(['   fval_quadprog - fval_di_mat = ', num2str(fval_qp - fval_mat), ...
    ', achieved with tolerance = ', num2str(tol)])
disp(['   fval_quadprog - fval_di_fun = ', num2str(fval_qp - fval_fun), ...
    ', achieved with tolerance = ', num2str(tol)])
disp(['   fval_di_mat   - fval_di_fun = ', num2str(fval_mat - fval_fun)])
