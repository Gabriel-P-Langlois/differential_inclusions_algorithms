function [x,d] = epgd_lsqnonneg(A,b,tol,x)
%   EPGD_LSQNONNEG    Differential inclusions iterative approach to solving
%                   the nonnegative least-squares (NNLS) problem, i.e.,
%                   projected gradient descent with exact line search
%                   using the minimal selection.
% 
%   This function computes an optimal solution to the NNLS problem
%                       
%       min_{x \in \Rn} (1/2)\|A*x - b\|_2^2   subject to x \geqslant 0
%
%   via differential inclusions, i.e., projected gradient descent with
%   exact line search using the minimal selection.
%
% -------------------------------------------------------------------------
%   INPUTS
%       A       -   (m x n)-dimensional matrix A
%       b       -   m-dimensional col data vector.
%       tol     -   (Optional) small number specifying 
%                   the tolerance (e.g., 1e-08). Default value
%                   is derived from the Frobenius norm of A
%                   and the relative size of b.
%       x       -   (Optional) Initial feasible point x >= 0.
%
%   OUTPUTS
%       x       -   n-dimensional solution vector to the NNLS problem.
%       d       -   m-dimensional col residual vector d = A*x - b.
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
%       % Compare MATLAB's default lsqnonneg vs epgd_lsqnonneg
%       tic
%       [x_default,rnorm,residual] = lsqnonneg(A,b);
%       time_default = toc;
%
%       tic
%       [x_epgd,d] = epgd_lsqnonneg(A,b);
%       time_epgd = toc;
%
%       % Compute the \ell_{\inf} error between the residuals A*x - b
%       % obtained from lsqnonneg and epgd_lsqnonneg.
%       abs_error_res = norm(-d - residual,'inf');
%
% -------------------------------------------------------------------------
%   NOTES
%       1) This is an iterative algorithm, not an exact or active-set one.
%
%       2) Each iteration makes two matrix-vector multiplications; this is
%       the dominant cost of the algorithm.
%
%       3) The stepsize is taken so as to not violate the feasibility
%       region and otherwise maximize the amount of progress made at each
%       iteration.
%
% -------------------------------------------------------------------------


%%  Preliminary checks
% Check for an acceptable number of input arguments
if nargin < 2
    error('epgd_lsqnonneg: Not enough input arguments.')
end
if ~iscolumn(b)
    error('epgd_lsqnonneg: The data vector b is not a column vector.');
end

% Check matrix and right hand side vector inputs have appropriate sizes
[m,n] = size(A);
if size(b,1) ~= m
    error(['epgd_lsqnonneg: The number of rows of A does not match the' ...
        'length of the column vector b.'])
end

% Check if the optional input arguments have been supplied.
if nargin < 3 || isempty(tol)
    tol = min(1e-08,100*eps*max(norm(A, 'fro'), 1)*max(norm(b), 1));
end
if nargin < 4 || isempty(x)
    if issparse(A)
        x = sparse(n,1);
    else
        x = zeros(n,1);
    end
end

% Check if x is feasible
if(min(x) < 0)
    x = max(0,x);
end

%% Algorithm
% Initialization
d = A*x - b;
eqset = x < tol;
eqcset = ~eqset;
At_times_d = (d.'*A).';
kmax = 50000;
xi = zeros(n,1);

% Main loop
for k = 1:1:kmax
    % Compute the descent direction:
    %   dms = xi - (Q*x0 - b), where xi = max(0,(Q*x0 - b))_{active}
    xi(eqset) = max(0,At_times_d(eqset));
    xi(~eqset) = 0;
    dms = xi - At_times_d;

    % Compute the feasibility timestep.
    delta_set = (eqcset) & (dms < 0);
    if(any(delta_set))
        tmp = x./abs(min(0,dms));
        delta_feas = min(tmp(delta_set));
    else
        delta_feas = inf;
    end

    % Compute the optimal timestep
    vec1 = A*dms;
    vec2 = A.'*vec1;
    delta_opt = (dms.'*dms) / (vec1.'*vec1);

    % Compute the effective timestep
    delta = min(delta_opt,delta_feas);

    % Check for convergence. If so, move on to the post-processing part.
    if(norm(dms,'inf') < tol)
        break;
    end

    % Update x and d.
    x = x + delta*dms;
    d = d + delta*vec1;
    At_times_d = At_times_d + delta*vec2;

    % Update the equicorrelation set
    eqset = x < tol;
    eqcset = ~eqset;
end


%% Postprocess the solution.
warning('off', 'MATLAB:lsqr:tooSmallTolerance')
[x(eqcset),~] = lsqr(A(:,eqcset),b,eps,1000,[],[],x(eqcset));
warning('on', 'MATLAB:lsqr:tooSmallTolerance')

x(eqset) = 0;
d = A*x-b;
end