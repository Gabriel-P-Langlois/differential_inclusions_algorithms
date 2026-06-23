%   TEST_QP     This script verifies that di_qp.m works correctly with
%               each supported NNLS solver, and compares against rapdhg_qp.
%
%               The quadratic program in question is given by
%               \min_{x \in \Rn} (1/2)*x'*Q*x + c'*x  subject to A*x <= b,
%               where Q is symmetric positive definite.
%
% -------------------------------------------------------------------------
%   FUNCTIONS THAT ARE TESTED
%       [x,fval] = quadprog(Q,c,A,b)
%           -   Default MATLAB solver for solving QP problems.
%
%       [x,~,fval,~] = di_qp(A,b,c,Q,x,tol,[],@hinges_lsqnonneg)
%           -   Differential inclusions with the method of hinges.
%
%       [x,~,fval,~] = di_qp(A,b,c,Q,x,tol,[],@epgd_lsqnonneg)
%           -   Differential inclusions with exact projected gradient descent.
%
%       [x,~,fval,~] = di_qp(A,b,c,Q,x,tol,[],@apgd_lsqnonneg)
%           -   Differential inclusions with accelerated projected gradient
%               descent.
%
%       [x,~,fval,~] = di_qp(A,b,c,Q,x,tol,[],@lbfgs_lsqnonneg)
%           -   Differential inclusions with LBFGS method.
%
%       [x,~,fval,~] = di_qp(A,b,c,Q,x,tol,[],@pcg_lsqnonneg)
%           -   Differential inclusions with PCG method.
%
% -------------------------------------------------------------------------


%% Preliminaries
rng(1);

% Ensure MATLAB's quadprog takes priority over MOSEK's shadowing version.
addpath(fullfile(matlabroot, 'toolbox', 'optim', 'optim'));

% Problem dimensions
m = 100;
n = 200;

% Generate a feasible interior point x0 with slack s > 0.
x0     = randn(n, 1);
A      = randn(m, n);
s      = abs(randn(m, 1)) + 1;
b      = A * x0 + s;

% Generate a random SPD matrix Q and cost vector c.
B = randn(n, n);
Q = B' * B + eye(n);
c = randn(n, 1);

% Boundary point via ray casting from x0.
d   = randn(n, 1);
Ad  = A * d;
idx = Ad > 0;
t   = min(s(idx) ./ Ad(idx));
x1  = x0 + t * d;


%% Test 1: Correctness.
tol = min(1e-08,100*eps*max(norm(A, 'fro'), 1)*max(norm(b), 1));
fprintf("QP Test: Correctness.\n\n")

% MATLAB's native quadprog function
tic
fprintf("Running MATLAB's quadprog...\n")
opts = optimoptions('quadprog', 'Display', 'off', ...
    'OptimalityTolerance', tol, 'Algorithm', 'interior-point-convex');
[x_qp, fval, ~, ~, lambda_qp] = quadprog(Q, c, A, b, [], [], [], [], [], opts);
time_quadprog = toc;
fprintf("Done.\n\n")

% Differential inclusions: method of hinges
tic
fprintf("Running di_qp with hinges_lsqnonneg...\n")
[x_hinges, p_hinges, fval_di_hinges, flag_hinges] = di_qp(A,b,c,Q,[],tol,[],@hinges_lsqnonneg);
time_di_hinges = toc;
fprintf("Done.\n\n")

% Differential inclusions: exact projected gradient descent
tic
fprintf("Running di_qp with epgd_lsqnonneg...\n")
[x_epgd, p_epgd, fval_di_epgd, flag_epgd] = di_qp(A,b,c,Q,[],tol,[],@epgd_lsqnonneg);
time_di_epgd = toc;
fprintf("Done.\n\n")

% Differential inclusions: accelerated projected gradient descent
tic
fprintf("Running di_qp with apgd_lsqnonneg...\n")
[x_apgd, p_apgd, fval_di_apgd, flag_apgd] = di_qp(A,b,c,Q,[],tol,[],@apgd_lsqnonneg);
time_di_apgd = toc;
fprintf("Done.\n\n")

% Differential inclusions: L-BFGS-B
tic
fprintf("Running di_qp with lbfgs_lsqnonneg...\n")
[x_lbfgs, p_lbfgs, fval_di_lbfgs, flag_lbfgs] = di_qp(A,b,c,Q,[],tol,[],@lbfgs_lsqnonneg);
time_di_lbfgs = toc;
fprintf("Done.\n\n")

% Differential inclusions: PCG
tic
fprintf("Running di_qp with pcg_lsqnonneg...\n")
[x_pcg, p_pcg, fval_di_pcg, flag_pcg] = di_qp(A,b,c,Q,[],tol,[],@pcg_lsqnonneg);
time_di_pcg = toc;
fprintf("Done.\n\n")



%% Residuals

Qx_c = @(x, p) Q*x + c + A'*p;

kkt_stat_qp    = norm(Qx_c(x_qp,    lambda_qp.ineqlin), inf);
kkt_pfeas_qp   = max(0, max(A*x_qp   - b));
fgap_qp        = lambda_qp.ineqlin' * (b - A*x_qp);

kkt_stat_hinges  = norm(Qx_c(x_hinges, p_hinges), inf);
kkt_pfeas_hinges = max(0, max(A*x_hinges - b));
fgap_hinges      = p_hinges' * (b - A*x_hinges);

kkt_stat_epgd    = norm(Qx_c(x_epgd, p_epgd), inf);
kkt_pfeas_epgd   = max(0, max(A*x_epgd - b));
fgap_epgd        = p_epgd' * (b - A*x_epgd);

kkt_stat_apgd    = norm(Qx_c(x_apgd, p_apgd), inf);
kkt_pfeas_apgd   = max(0, max(A*x_apgd - b));
fgap_apgd        = p_apgd' * (b - A*x_apgd);

kkt_stat_lbfgs   = norm(Qx_c(x_lbfgs, p_lbfgs), inf);
kkt_pfeas_lbfgs  = max(0, max(A*x_lbfgs - b));
fgap_lbfgs       = p_lbfgs' * (b - A*x_lbfgs);

kkt_stat_pcg     = norm(Qx_c(x_pcg, p_pcg), inf);
kkt_pfeas_pcg    = max(0, max(A*x_pcg - b));
fgap_pcg         = p_pcg' * (b - A*x_pcg);


%% Summary table

solvers = {'quadprog', 'hinges', 'epgd', 'apgd', 'lbfgs', 'pcg'};
times   = [time_quadprog, time_di_hinges, time_di_epgd, ...
           time_di_apgd,  time_di_lbfgs,  time_di_pcg];
fvals   = [fval, fval_di_hinges, fval_di_epgd, ...
           fval_di_apgd, fval_di_lbfgs, fval_di_pcg];
stats   = [kkt_stat_qp,   kkt_stat_hinges, kkt_stat_epgd, ...
           kkt_stat_apgd, kkt_stat_lbfgs,  kkt_stat_pcg];
pfeas   = [kkt_pfeas_qp,   kkt_pfeas_hinges, kkt_pfeas_epgd, ...
           kkt_pfeas_apgd, kkt_pfeas_lbfgs,  kkt_pfeas_pcg];
fgaps   = [fgap_qp,   fgap_hinges, fgap_epgd, ...
           fgap_apgd, fgap_lbfgs,  fgap_pcg];
flags   = [1, flag_hinges, flag_epgd, flag_apgd, flag_lbfgs, flag_pcg];

w = 85;
fprintf('\n%s\n', repmat('=', 1, w))
fprintf('Summary  (m = %d,  n = %d,  tol = %.2e)\n\n', m, n, tol)
fprintf('  %-10s  %8s  %15s  %10s  %10s  %10s  %4s\n', ...
        'Solver', 't (s)', 'fval', 'kkt_stat', 'kkt_pfeas', 'fgap', 'flag')
fprintf('  %s\n', repmat('-', 1, w-2))
for k = 1:length(solvers)
    fprintf('  %-10s  %8.3f  %+15.8e  %10.2e  %10.2e  %10.2e  %4d\n', ...
            solvers{k}, times(k), fvals(k), stats(k), pfeas(k), fgaps(k), flags(k))
end
fprintf('%s\n\n', repmat('=', 1, w))
