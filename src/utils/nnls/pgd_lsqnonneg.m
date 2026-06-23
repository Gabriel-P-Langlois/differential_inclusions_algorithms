function [xk,d,v] = pgd_lsqnonneg(A,b,tol,xk,v)
%   PGD_LSQNONNEG   Projected Gradient Descent iterative approach to
%                   solving the nonnegative least-squares (NNLS) problem.
% 
%   This function computes an optimal solution to the NNLS problem
%                       
%       min_{x \in \Rn} (1/2)\|A*x - b\|_2^2   subject to x \geqslant 0
%
%   via the (vanilla) projected gradient descent.
%
% -------------------------------------------------------------------------
%   INPUTS
%       A       -   (m x n)-dimensional matrix A
%       b       -   m-dimensional col data vector.
%       tol     -   (Optional) small number specifying 
%                   the tolerance (e.g., 1e-08). Default value
%                   is derived from the Frobenius norm of A
%                   and the relative size of b.
%       xk      -   (Optional) Initial feasible point xk >= 0.
%       v       -   (Optional) Initial estimate of the vector for computing
%                   the Lipschitz constant L.
%
%   OUTPUTS
%       xk      -   n-dimensional solution vector to the NNLS problem.
%       d       -   m-dimensional col residual vector d = A*xk - b.
%       v       -   Singular vector of A at the largest singular value.
%
% -------------------------------------------------------------------------
%   QUICK EXAMPLE
%       % Global parameters
%       rng(1);
%       tol = 1e-8;
%       m = 200;
%       n = 2000;
%
%       % Generate random Gaussian data
%       A = randn(m,n);
%       b = randn(m,1);
%
%       % Compare MATLAB's default lsqnonneg vs pgd_lsqnonneg
%       tic
%       [x_default,rnorm,residual] = lsqnonneg(A,b);
%       time_default = toc;
%
%       tic
%       [x_pgd,d] = pgd_lsqnonneg(A,b);
%       time_pgd = toc;
%
%       % Compute the \ell_{\inf} error between the residuals A*x - b
%       % obtained from lsqnonneg and pgd_lsqnonneg.
%       abs_error_res = norm(-d - residual,'inf');
%
% -------------------------------------------------------------------------
%   NOTES
%       1) This is an iterative algorithm, not an exact or active-set one.
%
%       2) Each iteration makes two matrix-vector multiplications; this is
%       the dominant cost of the algorithm.
%
% -------------------------------------------------------------------------


%%  Preliminary checks
% Check for an acceptable number of input arguments
if nargin < 2
    error('pgd_lsqnonneg: Not enough input arguments.')
end
if ~iscolumn(b)
    error('pgd_lsqnonneg: The data vector b is not a column vector.');
end

% Check matrix and right hand side vector inputs have appropriate sizes
[m,n] = size(A);
if size(b,1) ~= m
    error(['pgd_lsqnonneg: The number of rows of A does not match the' ...
        'length of the column vector b.'])
end

% Check if the optional input arguments have been supplied.
if nargin < 3 || isempty(tol)
    tol = min(1e-08,100*eps*max(norm(A, 'fro'), 1)*max(norm(b), 1));
end
if nargin < 4 || isempty(xk)
    if issparse(A)
        xk = sparse(n,1);
    else
        xk = zeros(n,1);
    end
end

% Check if x is feasible
if(min(xk) < 0)
    xk = max(0,xk);
end


%% Algorithm
% Initialization
d = A*xk - b;
At_times_d = A.'*d;
kmax = 200000;

% Estimate ||A||_{2} via MATLAB's normest function with warm start.
[L,v] = warm_normest(A,v,n);
tau = 1/L^2;

% Main loop
for k = 1:1:kmax
    % Iterates
    xkp = max(0,xk - tau*At_times_d);

    % Update residuals
    d = A*xkp - b;
    At_times_d = A.'*d;

    % Check for convergence. If so, move on to the post-processing part.
    if(norm(xkp-xk) < tol*norm(xk))
        break;
    end
    
    % Update for the next iterate
    xk = xkp;
end

% Postprocess xk to improve its quality via an iterative solver
[xk,d] = postprocess_sol(A, b, xk, xk>tol );
end


%% Utility functions

% Estimate the Lipschitz constant of the matrix A via MATLAB's normest
% function, which is here tailored to accept a warm start.
function [L,v] = warm_normest(A,v,n)
e = norm(v);
if(e < 1e-06)
    v = randn(n,1);
    e = norm(v);
end

cnt = 0;
v = v/e;
e0 = 0;
while(abs(e-e0) > 1e-06*e)
    e0 = e;
    Av = A*v;
    v = A.'*Av;
    normv = norm(v);
    e = normv/norm(Av);
    v = v/normv;
    cnt = cnt + 1;
    if(cnt > 50000)
        disp("The power method for computing L did not converge.")
    end
end
L = e;
end

% Postprocess the solution of the NNLS problem on its active set
% using MATLAB's LSQR solver.
function [xk,d] = postprocess_sol(A,b,xk,eqcset)
    warning('off', 'MATLAB:lsqr:tooSmallTolerance')
    xk(~eqcset) = 0;
    [xk(eqcset),~] = lsqr(A(:,eqcset),b,eps,1000,[],[],xk(eqcset));
    d = A*xk-b;
    warning('on', 'MATLAB:lsqr:tooSmallTolerance')
end