%   TEST_NNLS_SPARSE   This script verifies that epgd_lsqnonneg,
%                      apgd_lsqnonneg, and lbfgs_lsqnonneg work well
%                      with sparse data.
%
% -------------------------------------------------------------------------
%   FUNCTIONS THAT ARE TESTED
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


%% Test: Sparse data works for epgd_lsqnonneg
fprintf("NNLS Test (Sparsity): Testing epgd_lsqnonneg with sparse data.\n\n")
% Create a rectangular sparse matrix A (m > n overdetermined)
rng(1)
tol = 1e-10;
m = 10000; n = 5000; 
density = 0.15;

A = sprand(m, n, density); 
b = randn(m,1);

fprintf("Running PGD with exact line search...\n")
tic
[x_epgd,d_epgd] = epgd_lsqnonneg(A,b,tol,sparse(n,1));
fval_epgd = 0.5*norm(A*x_epgd - b)^2;
time_epgd = toc;
fprintf("Done.\n\n")


tic
fprintf("Running accelerated projected gradient descent...\n")
[x_apgd,d_apgd] = apgd_lsqnonneg(A,b,tol,sparse(n,1),[]);
fval_apgd = 0.5*norm(A*x_apgd - b)^2;
time_apgd = toc;
fprintf("Done.\n\n")

tic
fprintf("Running L-BFGS-B...\n")
[x_lbfgs,d_lbfgs] = lbfgs_lsqnonneg(A,b,tol,sparse(n,1));
fval_lbfgs = 0.5*norm(A*x_lbfgs - b)^2;
time_lbfgs = toc;
fprintf("Done.\n\n")