%   TEST_LP     This script verifies that di_lp.m works correctly with
%               each supported NNLS solver.
%
%               The linear program in question is given by
%               \min_{x \in \Rn} <c,x>  subject to A*x <= b.
%
% -------------------------------------------------------------------------
%   FUNCTIONS THAT ARE TESTED
%       [x,fval] = linprog(c,A,b)
%           -   Default MATLAB solver for solving LP problems.
%
%       [x,~,fval,~] = di_lp(A,b,c,x,tol,[],@hinges_lsqnonneg)
%           -   Differential inclusions with the method of hinges.
%
%       [x,~,fval,~] = di_lp(A,b,c,x,tol,[],@epgd_lsqnonneg)
%           -   Differential inclusions with exact projected gradient descent.
%
%       [x,~,fval,~] = di_lp(A,b,c,x,tol,[],@pgd_lsqnonneg)
%           -   Differential inclusions with projected gradient descent.
%
%       [x,~,fval,~] = di_lp(A,b,c,x,tol,[],@apgd_lsqnonneg)
%           -   Differential inclusions with accelerated projected gradient
%               descent.
%
%       [x,~,fval,~] = di_lp(A,b,c,x,tol,[],@lbfgs_lsqnonneg)
%           -   Differential inclusions with LBFGS method.
%
%       [x,~,fval,~] = di_lp(A,b,c,x,tol,[],@pcg_lsqnonneg)
%           -   Differential inclusions with pcg method.
%
% -------------------------------------------------------------------------


%% Preliminaries
% Parameters
rng(1);

% Generate the data
m = 500;
n = 1000;
x0     = randn(n, 1);
A      = randn(m, n);
s      = abs(randn(m, 1)) + 1;
b      = A * x0 + s;
lambda = abs(randn(m, 1)) + 1;  % strictly positive dual variable
c      = -A' * lambda;

% Boundary point via ray casting from x0
d  = randn(n, 1);
Ad = A * d;
idx = Ad > 0;
t  = min(s(idx) ./ Ad(idx));  % largest feasible step
x1 = x0 + t * d;


%% Test 1: Correctness.
tol = 100*eps*max(norm(A, 'fro'), 1)*max(norm(b), 1);
fprintf("LP Test: Correctness.\n\n")

% MATLAB's native linprog function
tic
fprintf("Running MATLAB's linprog...\n")
options = optimoptions('linprog','Display','none',...
    'OptimalityTolerance',tol,Algorithm="interior-point");
[x_linprog,fval] = linprog(c,A,b,[],[],[],[],options);
time_linprog = toc;
fprintf("Done.\n\n")

% Differential inclusions: method of hinges
tic
fprintf("Running di_lp with hinges_lsqnonneg " + ...
    "and boundary feasible point...\n")
[x_di_hinges,~,fval_di_hinges,~] = di_lp(A,b,c,x1,tol,[], ...
    @hinges_lsqnonneg);
time_di_hinges_lp = toc;
fprintf("Done.\n\n")

% Differential inclusions: exact projected gradient descent
tic
fprintf("Running di_lp with epgd_lsqnonneg " + ...
    "and boundary feasible point...\n")
[x_di_epgd,~,fval_di_epgd,~] = di_lp(A,b,c,x1,tol,[], @epgd_lsqnonneg);
time_di_epgd_lp = toc;
fprintf("Done.\n\n")

% Differential inclusions: vanilla projected gradient descent
tic
fprintf("Running di_lp with pgd_lsqnonneg " + ...
    "and boundary feasible point...\n")
[x_di_pgd,~,fval_di_pgd,~] = di_lp(A,b,c,x1,tol,[], @pgd_lsqnonneg);
time_di_pgd_lp = toc;
fprintf("Done.\n\n")

% Differential inclusions: accelerated projected gradient descent
tic
fprintf("Running di_lp with apgd_lsqnonneg and " + ...
    "boundary feasible point...\n")
[x_di_apgd,~,fval_di_apgd,~] = di_lp(A,b,c,x1,tol,[],@apgd_lsqnonneg);
time_di_apgd_lp = toc;
fprintf("Done.\n\n")

% Differential inclusions: L-BFGS-B
tic
fprintf("Running di_lp with lbfgs_lsqnonneg and " + ...
    "boundary feasible point...\n")
[x_di_lbfgs,~,fval_di_lbfgs,~] = di_lp(A,b,c,x1,tol,[],@lbfgs_lsqnonneg);
time_di_lbfgs_lp = toc;
fprintf("Done.\n\n")

% Differential inclusions: PCG
tic
fprintf("Running di_lp with pcg_lsqnonneg and " + ...
    "boundary feasible point...\n")
[x_di_pcg,~,fval_di_pcg,~] = di_lp(A,b,c,x1,tol,[],@pcg_lsqnonneg);
time_di_pcg_lp = toc;
fprintf("Done.\n\n")




%% Performance checks
disp(['   <c,x_linprog - x_di_hinges> = ', ...
    num2str(fval - fval_di_hinges), ', achieved with tolerance = ', ...
    num2str(tol)])

disp(['   <c,x_linprog - epgd> = ', ...
    num2str(fval - fval_di_epgd), ', achieved with tolerance = ', ...
    num2str(tol)])

disp(['   <c,x_linprog - pgd> = ', ...
    num2str(fval - fval_di_pgd), ', achieved with tolerance = ', ...
    num2str(tol)])

disp(['   <c,x_linprog - apgd> = ', ...
    num2str(fval - fval_di_apgd), ', achieved with tolerance = ', ...
    num2str(tol)])

disp(['   <c,x_linprog - lbfgs> = ', ...
    num2str(fval - fval_di_lbfgs), ', achieved with tolerance = ', ...
    num2str(tol)])

disp(['   <c,x_linprog - pcg> = ', ...
    num2str(fval - fval_di_pcg), ', achieved with tolerance = ', ...
    num2str(tol)])

fprintf('\n')
disp(['   Time linprog  = ', num2str(time_linprog),      ' s'])
disp(['   Time hinges   = ', num2str(time_di_hinges_lp), ' s'])
disp(['   Time epgd     = ', num2str(time_di_epgd_lp),   ' s'])
disp(['   Time pgd      = ', num2str(time_di_pgd_lp),    ' s'])
disp(['   Time apgd     = ', num2str(time_di_apgd_lp),   ' s'])
disp(['   Time lbfgs    = ', num2str(time_di_lbfgs_lp),  ' s'])
disp(['   Time pcg      = ', num2str(time_di_pcg_lp),    ' s'])