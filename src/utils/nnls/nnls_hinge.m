function [x,d] = nnls_hinge(A,b,tol,eqset)
% NNLS_HINGE        This function computes the nonnegative LSQ problem
%                   min_{x>=0} ||A*x-b||_{2}^{2}
%                   using the ``Method of Hinges" presented in
%                   ``A Simple New Algorithm for Quadratic Programming 
%                   with Applications in Statistics" by Mary C. Meyers.
%
% INPUT:
%   A           -   (m x n)-dimensional matrix A
%   b           -   m-dimensional col data vector.
%   tol         -   (Optional) Small number specifying the tolerance. 
%                   If unsupplied, set tol = 1e-08;
%   eqset       -   (Optional) Initial active set. If unsupplied, set
%                   to the empty set, e.g., eqset = false(n,1).
%
% OUTPUT:
%   x   -   n-dimemsional solution vector to the NNLS problem.
%   d   -   m-dimensional residual vector d = A*x-b


% Initial checks
[~, n] = size(A);
if(nargin < 3)
    tol = 1e-08;
    eqset = false(n,1);
elseif(nargin ~= 4)
    eqset = false(n,1);
end

% Check if the least-squares solution is feasible. If so, we are done.
u = A(:,eqset)\b;
if(min(u) >= -tol)
    d = A(:,eqset)*u - b;
    x = u;
    return;
end

% Else, compute the solution via the method of hinges.
while(true)
    if(min(u) < -tol)       % Check if the primal constraint is violated.
        x = zeros(n,1);
        x(eqset) = u;
        [~,I] = min(x);
        eqset(I) = false;
    else                    % Check if the dual constraint is violated.
        theta = A(:,eqset)*u;
        rho = b - theta;
        [val,I] = max((rho.'*A));

        if(val > tol)
            eqset(I) = true;
        else
            break;          % STOP; all constraints are satisfied.
        end
    end

    % Compute the solution to A(:,eqset)*u = b and continue.
    u = A(:,eqset)\b;
end

% Return the solution vector x and the residual d = A*x-b
x = zeros(n,1); x(eqset) = u;
d = -rho;
end