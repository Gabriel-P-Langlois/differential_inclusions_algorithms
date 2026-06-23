function [x,d] = hinges_lsqnonneg(A,b,tol,x0)
%   HINGES_LSQNONNEG    The ``method of hinges" algorithm for solving the
%                       nonnegative least-squares (NNLS) problem, due to
%                       Mary Meyers in the paper ``A Simple New Algorithm 
%                       for Quadratic Programming with Applications 
%                       in Statistics".
% 
%   This function computes an optimal solution to the NNLS problem
%                       
%       min_{x \in \Rn} (1/2)\|A*x - b\|_2^2   subject to x \geqslant 0
%
%   via the method of hinges algorithm due to Mary Meyers. The
%   implementation below was written by Gabriel P. Langlois.
%
% -------------------------------------------------------------------------
%   INPUTS
%       A       -   (m x n)-dimensional matrix A
%       b       -   m-dimensional col data vector.
%       tol     -   (Optional) small number specifying
%                   the tolerance (e.g., 1e-08). Default value
%                   is derived from the Frobenius norm of A
%                   and the relative size of b.
%       x0      -   (Optional) Initial feasible point x0 >= 0.
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
%       % Compare MATLAB's default lsqnonneg vs hinges_lsqnonneg
%       tic
%       [x_default,rnorm,residual] = lsqnonneg(A,b);
%       time_default = toc;
%
%       tic
%       [x_hinges,d] = hinges_lsqnonneg(A,b);
%       time_hinges = toc;
%
%       % Compute the \ell_{\inf} error between the residuals A*x - b
%       % obtained from lsqnonneg and hinges_lsqnonneg.
%       abs_error_res = norm(-d - residual,'inf');
%
% -------------------------------------------------------------------------
%   NOTES
%       The hinge_lsqnonneg implementation starts from the initial 
%       active [n]. This is in contrast with MATLAB's lsqnonneg method, 
%       which starts from the empty set. Expect significant changes in
%       performance between the two functions in general just from that
%       alone. In applications where it is expected that the NNLS optimal
%       solution is active on almost all of [n], then hinges_lsqnonneg will
%       likely outperform the default lsqnonneg function.
%
% -------------------------------------------------------------------------


%%  Preliminary checks
warning("off")

% Check if the tolerance was supplied by the user.
if nargin < 3 || isempty(tol)
    tol = min(1e-08,100*eps*max(norm(A, 'fro'), 1)*max(norm(b), 1));
end

% Check if the LSQ solution is admissible. If so, return it.
if(~issparse(A))
    u = A\b;
else
    [u,~] = lsqr(A,b,eps,5000);
end
if(min(u) > -tol)
    x = u; d = A*x-b;
    return
end


%%  Compute the NNLS solutions via Meyers algorithm
[~, n] = size(A);
active_set = true(n,1);
while(true)
    % Check if the primal constraint is violated. If so, mark
    % the violating element. Else, compute the residual b - A*u.
    if(min(u) <= -tol)
        x = zeros(n,1);
        x(active_set) = u;
        [~,I] = min(x);
        active_set(I) = false;
    else
        theta = A(:,active_set)*u;
        theta = b - theta;
        Atop_times_rho = (theta.'*A);
        [val,I] = max(Atop_times_rho);

        % Check if the dual constraint holds. If so, EXIT. Otherwise,
        % mark the corresponding columns to be added to the active set.
        if(val < tol)
            break;
        else
            active_set(I) = true;
        end
    end

    % Compute LSQ solution to A(:,active_set)*u = b.
    if(~issparse(A))
        u = A(:,active_set)\b;
    else
        [u,~] = lsqr(A(:,active_set),b,eps,5000);
    end
end
x = zeros(n,1); x(active_set) = u;
d = -theta;
warning("on")
end