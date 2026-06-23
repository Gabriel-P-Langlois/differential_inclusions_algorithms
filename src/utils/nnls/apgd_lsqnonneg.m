function [xk,d,v] = apgd_lsqnonneg(A,b,tol,xk,v)
%   APGD_LSQNONNEG  Accelerated Projected Gradient Descent iterative
%                   approach to solving the nonnegative least-squares
%                   (NNLS) problem.
% 
%   This function computes an optimal solution to the NNLS problem
%                       
%       min_{x \in \Rn} (1/2)\|A*x - b\|_2^2   subject to x \geqslant 0
%
%   via the accelerated projected gradient descent \w restart. This is 
%   algorithm 4.2 of IMPROVING ``FAST ITERATIVE SHRINKAGE-THRESHOLDING
%   ALGORITHM"": FASTER, SMARTER, AND GREEDIER by JINGWEI LIANG\dagger, 
%   TAO LUO\ddagger AND CAROLA-BIBIANE SCHOENLIEB with p=q=1, r=4, \xi = 1.
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
%       % Compare MATLAB's default lsqnonneg vs apgd_lsqnonneg
%       tic
%       [x_default,rnorm,residual] = lsqnonneg(A,b);
%       time_default = toc;
%
%       tic
%       [x_apgd,d] = apgd_lsqnonneg(A,b);
%       time_apgd = toc;
%
%       % Compute the \ell_{\inf} error between the residuals A*x - b
%       % obtained from lsqnonneg and apgd_lsqnonneg.
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
    error('apgd_lsqnonneg: Not enough input arguments.')
end
if ~iscolumn(b)
    error('apgd_lsqnonneg: The data vector b is not a column vector.');
end

% Check matrix and right hand side vector inputs have appropriate sizes
[m,n] = size(A);
if size(b,1) ~= m
    error(['apgd_lsqnonneg: The number of rows of A does not match the' ...
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
if nargin < 5 || isempty(v)
    v = randn(n,1);
end

% Check if x is feasible
if(min(xk) < 0)
    xk = max(0,xk);
end


%% Algorithm
% Initialization
Atb = -A.'*b;
kmax = 50000;

% Estimate ||A||_{2} via MATLAB's normest function with warm start.
[L,v] = warm_normest(A,v,n);
tau = 1/L^2;

% Main loop
xkm = xk;
tk = 1.0;
for k = 1:1:kmax
    % Iterations
    tkp = 0.5*(1 + sqrt(1 + 4*tk^2));
    betak = (tk-1)/tkp;
    yk = xk + betak*(xk-xkm);

    d = A*yk; 
    Atd = A.'*d;
    xkp = max(0,yk - tau*(Atd + Atb));

    % Check if restarting is needed
    if((yk-xkp).'*(xkp-xk) > 0)
        yk = xk;
        d = A*yk; 
        Atd = A.'*d;
        xkp = max(0,yk - tau*(Atd + Atb));
        tkp = 1.0;
    end

    % Check for convergence. If so, move on to the postprocessing part.
    if(norm(xkp-xk) < tol*(1+norm(xkp)))
        xk = xkp;
        break;
    end

    % Update for the next iterate
    xkm = xk;
    xk  = xkp;
    tk  = tkp;
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
    normAv = norm(Av);
    v = A.'*Av;
    normv = norm(v);
    e = normv/normAv;
    v = v/normv;
    cnt = cnt + 1;
    if(cnt > 50000)
        disp("The power method for computing L did not converge.")
        break;
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