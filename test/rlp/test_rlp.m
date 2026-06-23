%   TEST_RLP    This script verifies that di_rlp.m works correctly with
%               each supported NNLS solver.
%
%               The regularized linear program in question is given by
%               \min_{x \in \Rn} (t/2)*\|x\|^2 + <c,x>
%               subject to A*x <= b,  with t > 0.
%
% -------------------------------------------------------------------------
%   FUNCTIONS THAT ARE TESTED
%       [x,fval] = quadprog(t*eye(n),c,A,b)
%           -   Default MATLAB solver for solving QP problems.
%
%       [x,p,fval,~] = di_rlp(A,b,c,t,x,tol,[],@hinges_lsqnonneg)
%           -   Differential inclusions with the method of hinges.
%
%       [x,p,fval,~] = di_rlp(A,b,c,t,x,tol,[],@epgd_lsqnonneg)
%           -   Differential inclusions with exact projected gradient descent.
%
%       [x,p,fval,~] = di_rlp(A,b,c,t,x,tol,[],@apgd_lsqnonneg)
%           -   Differential inclusions with accelerated projected gradient
%               descent.
%
%       [x,p,fval,~] = di_rlp(A,b,c,t,x,tol,[],@lbfgs_lsqnonneg)
%           -   Differential inclusions with the L-BFGS method.
%
%       [x,p,fval,~] = di_rlp(A,b,c,t,x,tol,[],@pcg_lsqnonneg)
%           -   Differential inclusions with the PCG method.
%
% -------------------------------------------------------------------------

% Ensure MATLAB's quadprog takes priority over MOSEK's shadowing version.
addpath(fullfile(matlabroot, 'toolbox', 'optim', 'optim'));


%% Preliminaries
% Parameters
rng(1);
t_reg = 1.0;    % regularization parameter (named t_reg to avoid clash below)

% Generate the data
m = 400;
n = 400;
x0     = randn(n, 1);
A      = randn(m, n);
s      = abs(randn(m, 1)) + 1;
b      = A * x0 + s;
lambda = abs(randn(m, 1)) + 1;  % strictly positive dual variable
c      = -A' * lambda;

% Boundary point via ray casting from x0
d   = randn(n, 1);
Ad  = A * d;
idx = Ad > 0;
tau = min(s(idx) ./ Ad(idx));   % largest feasible step
x1  = x0 + tau * d;


%% Test 1: Correctness.
tol = 100*eps*max(norm(A, 'fro'), 1)*max(norm(b), 1);
fprintf("RLP Test: Correctness.\n\n")

% MATLAB's native quadprog function (interior-point, default)
tic
fprintf("Running MATLAB's quadprog...\n")
opts_qp = optimoptions('quadprog', 'Display', 'none');
[x_qp, fval_qp] = quadprog(t_reg*eye(n), c, A, b, [], [], [], [], [], opts_qp);
time_quadprog = toc;
fprintf("Done.\n\n")

% Differential inclusions: method of hinges
tic
fprintf("Running di_rlp with hinges_lsqnonneg...\n")
[x_di_hinges, p_di_hinges, fval_di_hinges, ~] = ...
    di_rlp(A, b, c, t_reg, [], tol, [], @hinges_lsqnonneg);
time_di_hinges = toc;
fprintf("Done.\n\n")

% Differential inclusions: exact projected gradient descent
tic
fprintf("Running di_rlp with epgd_lsqnonneg...\n")
[x_di_epgd, p_di_epgd, fval_di_epgd, ~] = ...
    di_rlp(A, b, c, t_reg, [], tol, [], @epgd_lsqnonneg);
time_di_epgd = toc;
fprintf("Done.\n\n")

% Differential inclusions: accelerated projected gradient descent
tic
fprintf("Running di_rlp with apgd_lsqnonneg...\n")
[x_di_apgd, p_di_apgd, fval_di_apgd, ~] = ...
    di_rlp(A, b, c, t_reg, [], tol, [], @apgd_lsqnonneg);
time_di_apgd = toc;
fprintf("Done.\n\n")

% Differential inclusions: L-BFGS
tic
fprintf("Running di_rlp with lbfgs_lsqnonneg...\n")
[x_di_lbfgs, p_di_lbfgs, fval_di_lbfgs, ~] = ...
    di_rlp(A, b, c, t_reg, [], tol, [], @lbfgs_lsqnonneg);
time_di_lbfgs = toc;
fprintf("Done.\n\n")

% Differential inclusions: PCG
tic
fprintf("Running di_rlp with pcg_lsqnonneg...\n")
[x_di_pcg, p_di_pcg, fval_di_pcg, ~] = ...
    di_rlp(A, b, c, t_reg, [], tol, [], @pcg_lsqnonneg);
time_di_pcg = toc;
fprintf("Done.\n\n")


%% Performance checks
fprintf("Primal objective gap  |fval_quadprog - fval_di|:\n\n")
disp(['   hinges : ', num2str(fval_qp - fval_di_hinges)])
disp(['   epgd   : ', num2str(fval_qp - fval_di_epgd)])
disp(['   apgd   : ', num2str(fval_qp - fval_di_apgd)])
disp(['   lbfgs  : ', num2str(fval_qp - fval_di_lbfgs)])
disp(['   pcg    : ', num2str(fval_qp - fval_di_pcg)])
fprintf("\n")

fprintf("KKT checks  (stationarity | dual feas | compl. slack):\n\n")
kkt = @(x,p) [norm(t_reg*x + c + A'*p, inf), ...
               max(0, -min(p)), ...
               norm(p .* (b - A*x), inf)];

res = kkt(x_di_hinges, p_di_hinges);
fprintf("   hinges : %e  %e  %e\n", res(1), res(2), res(3))
res = kkt(x_di_epgd, p_di_epgd);
fprintf("   epgd   : %e  %e  %e\n", res(1), res(2), res(3))
res = kkt(x_di_apgd, p_di_apgd);
fprintf("   apgd   : %e  %e  %e\n", res(1), res(2), res(3))
res = kkt(x_di_lbfgs, p_di_lbfgs);
fprintf("   lbfgs  : %e  %e  %e\n", res(1), res(2), res(3))
res = kkt(x_di_pcg, p_di_pcg);
fprintf("   pcg    : %e  %e  %e\n", res(1), res(2), res(3))
fprintf("\n")

fprintf("Wall-clock time (seconds):\n\n")
fprintf("   quadprog : %.4f\n", time_quadprog)
fprintf("   hinges   : %.4f\n", time_di_hinges)
fprintf("   epgd     : %.4f\n", time_di_epgd)
fprintf("   apgd     : %.4f\n", time_di_apgd)
fprintf("   lbfgs    : %.4f\n", time_di_lbfgs)
fprintf("   pcg      : %.4f\n", time_di_pcg)
