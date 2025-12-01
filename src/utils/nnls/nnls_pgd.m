function [x,d] = nnls_pgd(A,b,x,kmax,tol,eta)
% NNLS_PGD          This function computes a solution to the
%                   NNLS problem min_{x>= 0} ||Ax-b||_{2}^{2}
%                   using the projected gradient descent metod.
%
% INPUT:
%   A           - (m x n)-dimensional matrix A.
%   b           - m-dimensional vector b.
%   x           - n-dimensional starting feasible point
%   kmax        - Maximum number of projected steps allowed.
%                 Default value is 50000.
%   tol         - positive number: tolerance before convergence
%                 Default value is 1e-08.
%   eta         - Fixed stepsize to use. Default is 2/norm(A)^2.
%
% OUTPUT:
%   x           - n-dimensional solution of the NNLS problem
%   d           - m-dimensional residual vector Ax-b
%   iters       - number of iterations performed by the algorithm


%% Initialization
if(nargin < 4)
    kmax = 50000;
    tol = 1e-08;
    eta = 2/norm(A)^2;
elseif(nargin < 5)
    tol = 1e-08;
    eta = 2/norm(A)^2;
elseif(nargin < 6)
    eta = 2/norm(A)^2;
end


%% Projected Gradient Descent with constraints x >=0.
for k=1:1:kmax
    d = A*x-b;
    z = A.'*d;
    y = x - eta*z;

    if(norm(x-max(0,y),'inf')/norm(x,'inf') < tol)
        break;
    end
    x = max(0,y);
end
if(k == kmax)
    warning("PGD: Maximum number of iterates reached!")
end
end
