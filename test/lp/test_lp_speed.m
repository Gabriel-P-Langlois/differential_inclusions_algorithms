%   TEST_LP_SPEED   This script tests the running speed of di_lp.m with
%                   each supported NNLS solver.
%
%                   The linear program in question is given by
%                   \min_{x \in \Rn} <c,x>  subject to A*x <= b.
%
% -------------------------------------------------------------------------
%   FUNCTIONS THAT ARE TESTED
%       [x,fval] = linprog(c,A,b)
%           -   Default MATLAB solver for solving LP problems. Different
%               algorithms are used.
%
%       [x,~,fval,~] = di_lp(A,b,c,x,tol,[],@apgd_lsqnonneg)
%           -   Differential inclusions with accelerated
%               projected gradient descent.
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


%% Test 1: Speed.
% Generate the data
m = 1000;
n = 2000;
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

tol = min(1e-08,100*eps*max(norm(A, 'fro'), 1)*max(norm(b), 1));
fprintf("LP Test: Speed.\n\n")

% MATLAB's native linprog function -- interior point
tic
fprintf("Running MATLAB's linprog (interior point)...\n")
options = optimoptions('linprog','Display','none', ...
    Algorithm="interior-point");
[x_linprog_ip,fval_ip] = linprog(c,A,b,[],[],[],[],options);
time_linprog_ip = toc;
fprintf("Done.\n\n")

% % Differential inclusions: method of hinges
% tic
% fprintf("Running di_lp with hinges_lsqnonneg...\n")
% [x_di_hinges,~,fval_di_hinges,~] = di_lp(A,b,c,[],tol,[], ...
%     @hinges_lsqnonneg);
% time_di_hinges_lp = toc;
% fprintf("Done.\n\n")
% 
% % Differential inclusions: exact projected gradient descent
% tic
% fprintf("Running di_lp with epgd_lsqnonneg...\n")
% [x_di_epgd,~,fval_di_epgd,~] = di_lp(A,b,c,[],tol,[],@epgd_lsqnonneg);
% time_di_epgd_lp = toc;
% fprintf("Done.\n\n")
% 
% % Differential inclusions: vanilla projected gradient descent
% tic
% fprintf("Running di_lp with pgd_lsqnonneg...\n")
% [x_di_pgd,~,fval_di_pgd,~] = di_lp(A,b,c,[],tol,[],@pgd_lsqnonneg);
% time_di_pgd_lp = toc;
% fprintf("Done.\n\n")

% Differential inclusions: accelerated projected gradient descent
tic
fprintf("Running di_lp with apgd_lsqnonneg...\n")
[x_di_apgd,~,fval_di_apgd,~] = di_lp(A,b,c,[],tol,[],@apgd_lsqnonneg);
time_di_apgd_lp = toc;
fprintf("Done.\n\n")

% Differential inclusions: L-BFGS-B
tic
fprintf("Running di_lp with lbfgs_lsqnonneg...\n")
[x_di_lbfgs,~,fval_di_lbfgs,~] = di_lp(A,b,c,[],tol,[],@lbfgs_lsqnonneg);
time_di_lbfgs_lp = toc;
fprintf("Done.\n\n")

% Differential inclusions: PCG
tic
fprintf("Running di_lp with pcg_lsqnonneg...\n")
[x_di_pcg,~,fval_di_pcg,~] = di_lp(A,b,c,[],tol,[],@pcg_lsqnonneg);
time_di_pcg_lp = toc;
fprintf("Done.\n\n")



%% Time checks
fprintf(['(lingprog ip) fval = ', num2str(fval_ip), ...
    ', time: ',num2str(time_linprog_ip), ' seconds.\n\n'])

% fprintf(['(hinges) fval = ', num2str(fval_di_hinges), ...
%     ', time: ',num2str(time_di_hinges_lp), ' seconds.\n\n'])
% 
% fprintf(['(epgd) fval = ', num2str(fval_di_epgd), ...
%     ', time: ',num2str(time_di_epgd_lp), ' seconds.\n\n'])
% 
% fprintf(['(pgd) fval = ', num2str(fval_di_pgd), ...
%     ', time: ',num2str(time_di_pgd_lp), ' seconds.\n\n'])

fprintf(['(apgd) fval = ', num2str(fval_di_apgd), ...
    ', time: ',num2str(time_di_apgd_lp), ' seconds.\n\n'])

fprintf(['(lbfgs) fval = ', num2str(fval_di_lbfgs), ...
    ', time: ',num2str(time_di_lbfgs_lp), ' seconds.\n\n'])

fprintf(['(pcg) fval = ', num2str(fval_di_pcg), ...
    ', time: ',num2str(time_di_pcg_lp), ' seconds.\n\n'])