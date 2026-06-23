%   TEST_NNLS   This script verifies that all the implemented NNLS
%               algorithms work correctly.
%
% -------------------------------------------------------------------------
%   FUNCTIONS THAT ARE TESTED
%       [x,rnorm,residual] = lsqnonneg(A,b) 
%           -   Default MATLAB solver.
%
%       [x,d] = hinges_lsqnonneg(A,b)        
%           -   Basic implementation of the method of hinges algorithm.
%               Note: d = - residual; norm(d,2) = rnorm;
%
%       [x,d] = epgd_lsqnonneg(A,b)
%           -   Basic implementation of the differential inclusions
%               approach, i.e., projected gradient descent with exact
%               line search.
%
%       [xk,d,v] = pgd_lsqnonneg(A,b)
%           -   Basic implementation of projected gradient descent.
%
%       [xk,d,v] = apgd_lsqnonneg(A,b)
%           -   Basic implementation of accelerated
%               projected gradient descent.
%
%       [x,d] = lbfgs_lsqnonneg(A,b)
%           -   Basic implementation of the lbfgs method for NNLS.
%
%       [x,d] = pcg_lsqnonneg(A,b)
%           -   Basic implementation of the projected conjugate gradient
%               method for NNLS.
%
% -------------------------------------------------------------------------


%% Preliminaries
rng(1);


%%  Test 1: Correctness.
% Parameters
m = 100; n = 50;
tol = 1e-08;

% Generate random Gaussian data
A = randn(m,n);
b = randn(m,1);

% Run the tests
fprintf("NNLS Test I: Correctness.\n\n")
tic
fprintf("Running MATLAB's lsqnonneg...\n")
[x_default,rnorm,residual] = lsqnonneg(A,b);
fval_lsqnonneg = 0.5*norm(A*x_default-b)^2;
time_default = toc;
fprintf("Done.\n\n")

tic
fprintf("Running the method of hinges...\n")
[x_hinges,d_hinges] = hinges_lsqnonneg(A,b,tol);
fval_hinges = 0.5*norm(A*x_hinges-b)^2;
time_hinge = toc;
fprintf("Done.\n\n")

tic
fprintf("Running PGD with exact line search...\n")
[x_epgd,d_epgd] = epgd_lsqnonneg(A,b,tol);
fval_epgd = 0.5*norm(A*x_epgd-b)^2;
time_epgd = toc;
fprintf("Done.\n\n")

tic
fprintf("Running vanilla PGD...\n")
[x_pgd,d_pgd] = pgd_lsqnonneg(A,b,tol,[],[]);
fval_pgd = 0.5*norm(A*x_pgd-b)^2;
time_pgd = toc;
fprintf("Done.\n\n")

tic
fprintf("Running accelerated PGD...\n")
[x_apgd,d_apgd] = apgd_lsqnonneg(A,b,tol,[],[]);
fval_apgd = 0.5*norm(A*x_apgd-b)^2;
time_apgd = toc;
fprintf("Done.\n\n")

tic
fprintf("Running L-BFGS-B...\n")
[x_lbfgs,d_lbfgs] = lbfgs_lsqnonneg(A,b,tol);
fval_lbfgs = 0.5*norm(A*x_lbfgs-b)^2;
time_lbfgs = toc;
fprintf("Done.\n\n")

tic
fprintf("Running Projection Conjugate Gradient method......\n")
[x_pcg,d_pcg] = pcg_lsqnonneg(A,b,tol);
fval_pcg = 0.5*norm(A*x_pcg-b)^2;
time_pcg = toc;
fprintf("Done.\n\n")


% Compute the \ell_{\inf} error between the residuals A*x - b
% obtained from lsqnonneg and hinges_lsqnonneg.
abs_error_res_hinge = norm(-d_hinges + residual,'inf');

% Compute the \ell_{\inf} error between the residuals A*x - b
% obtained from lsqnonneg and epgd_lsqnonneg.
abs_error_res_epgd = norm(-d_epgd + residual,'inf');

% Compute the \ell_{\inf} error between the residuals A*x - b
% obtained from lsqnonneg and pgd_lsqnonneg.
abs_error_res_pgd = norm(-d_pgd + residual,'inf');

% Compute the \ell_{\inf} error between the residuals A*x - b
% obtained from lsqnonneg and apgd_lsqnonneg.
abs_error_res_apgd = norm(-d_apgd + residual,'inf');

% Compute the \ell_{\inf} error between the residuals A*x - b
% obtained from lsqnonneg and lbfgs_lsqnonneg.
abs_error_res_lbfgs = norm(-d_lbfgs + residual,'inf');

% Compute the \ell_{\inf} error between the residuals A*x - b
% obtained from lsqnonneg and pgc_lsqnonneg.
abs_error_res_pcg = norm(-d_pcg + residual,'inf');

fprintf("The following numbers should be < tol: \n")
disp([abs_error_res_hinge, abs_error_res_pgd])

fprintf("The following numbers should be < tol: \n")
disp([abs_error_res_hinge, abs_error_res_epgd])

fprintf("The following numbers should be < tol: \n")
disp([abs_error_res_hinge, abs_error_res_apgd])

fprintf("The following numbers should be < tol: \n")
disp([abs_error_res_hinge, abs_error_res_lbfgs])

fprintf("The following numbers should be < tol: \n")
disp([abs_error_res_hinge, abs_error_res_pcg])