%   TEST_NNLS_DENSE    This script verifies that epgd_lsqnonneg,
%                      apgd_lsqnonneg, and lbfgs_lsqnonneg work well
%                      with dense data.
%
% -------------------------------------------------------------------------
%   FUNCTIONS THAT ARE TESTED
%
%       [x,d] = epgd_lsqnonneg(A,b)
%           -   Basic implementation of the differential inclusions
%               approach, i.e., projected gradient descent with exact
%               line search.
%
%       [xk,d,v] = apgd_lsqnonneg(A,b)
%           -   Basic implementation of accelerated projected gradient
%               descent.
%
%       [x,d] = lbfgs_lsqnonneg(A,b)
%           -   Basic implementation of the lbfgs method for NNLS.
%
% -------------------------------------------------------------------------


%% Test: Dense data works for epgd_lsqnonneg
fprintf("NNLS Test: Testing epgd_lsqnonneg with dense data.\n\n")
% Create a rectangular dense matrix A (m > n overdetermined)
rng(1)
tol = 1e-10;
m = 2000; n = 1000; 

A = rand(m, n); 
b = sum(A, 2); % Expected exact solution x is a vector of ones.

tic
fprintf("Running exact projected gradient descent...\n")
[x_epgd,d_epgd] = epgd_lsqnonneg(A,b,[],zeros(n,1));
fval_epgd = 0.5*norm(A*x_epgd - b)^2;
time_epgd = toc;
fprintf("Done.\n\n")

tic
fprintf("Running accelerated projected gradient descent...\n")
[x_apgd,d_apgd] = apgd_lsqnonneg(A,b,[],zeros(n,1),[]);
fval_apgd = 0.5*norm(A*x_apgd - b)^2;
time_apgd = toc;
fprintf("Done.\n\n")

tic
fprintf("Running L-BFGS-B...\n")
[x_lbfgs,d_lbfgs] = lbfgs_lsqnonneg(A,b,[],zeros(n,1));
fval_lbfgs = 0.5*norm(A*x_lbfgs - b)^2;
time_lbfgs = toc;
fprintf("Done.\n\n")