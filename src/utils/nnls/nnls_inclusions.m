function [x,r,iters] = nnls_inclusions(A,b,x,tol)
% NNLS_INCLUSIONS   This function computes the solution to the 
%                   NNLS problem min_{x>= 0} ||Ax-b||_{2}^{2}
%                   by integrating approximately its associated
%                   differential inclusions.
%
%                   The projected dynamical system is integrated via a
%                   simple Euler method.
%
%                   NOTE: Empirically works best for overdetermined matrix
%                   A. If underdetermined, use the dual problem instead.
%                   E.g., invoke nnls_inclusions(inv(A*A.')*A,-b,p,tol)
%                   and extract the results accordingly.
%
% INPUT:
%   A           - (m x n)-dimensional matrix A.
%   b           - m-dimensional vector b.
%   x           - n-dimensional starting (strictly) feasible point
%   tol         - positive number: tolerance before convergence
%                 Default value is 1e-08.
%
% OUTPUT:
%   x           - n-dimensional solution of the NNLS problem
%   r           - m-dimensional residual vector Ax-b
%   iters       - number of iterations performed by the algorithm


%% Initialization
if(nargin < 4)
    tol = 1e-08;
end
[~,n] = size(A); 
iters = 0;
r = A*x - b;

% Check if b == 0. If so, solution is zero.
bnorm = norm(b,'inf');
if(bnorm < tol)
    x = zeros(n,1);
    r = -b;
    return;
end

% Check if the LSQ solution is valid. If so, return it.
u = A\b;
if(min(u) > -tol)
    x = u;
    r = A*x - b;
    return;
end


%% Main algorithm
while(true)
    iters = iters + 1;
    % Compute current active set.
    active = (abs(x) <= tol);

    % Compute the descent direction
    d = calc_descent(A,r,active,n);

    % Compute the descent times.
    % 1. ``Optimal" descent time from the constraints.
    pos_set = ~active;
    delta_1 = inf;
    if(any(pos_set))
        tmp1 = x(pos_set);
        tmp2 = max(0,-d(pos_set));
        delta_1 = min(tmp1./tmp2);
    end

    % 2. Descent time from the Euler method.
    term1 = norm(d,2);
    term2 = A*d;
    term3 = norm(term2,2);
    delta_2 = (term1/term3)^2; % Can be singular...

    % 3. Pick smallest of the two descent times.
    delta = min(delta_1,delta_2);
    
    % Check for convergence
    check = norm(d,'inf')/bnorm;
    if(check <= tol)
        % Active set has been identified. Compute the corresponding LSQ sol
        x = zeros(n,1); x(~active) = A(:,~active)\b;
        r = A*x - b;
        break;
    end

    % Update
    x = x + delta*d; r = r + delta*term2;
end
end


%% Helper function
function d = calc_descent(A,r,eqset,n)
    z = A.'*r;
    if(~isempty(eqset))
        u = zeros(n,1); u(eqset) = max(0,z(eqset));
        d = u - z;
    else
        d = -z;
    end
end