function x = ldp_incl_solver(A,b,v,x)
% LDP_INCL_SOLVER    This function computes the exact solution of the
%                   least distance program
%
%                   min_{x \in \Rn} 0.5*||x - v||_{2}^{2}
%                                   subject to Ax=b,
%
%                   by integrating the differential inclusions
%                   arising from this problem.
%
% INPUT:
%   A       -  (m x n)-dimensional matrix.
%   b       -  m-dimensional vector
%   v       -  n-dimensional vector
%   x       -  (Optional) Starting point. If none is supplied, will default
%              to solving the Phase I problem
%              min_{x \in \Rn} ||x||_{1} subject to Ax <= b,
%
%
% OUTPUT:
%   x       - Optimal solution to the problem, if it exists. If not,
%             return empty.

% Notes: The matrix A must be of full row rank. This algorithm is nothing
%        new


%% Preliminary checks
% Check if the arguments are correct
if(nargin < 3)
    error("Invalid input: Supplied too few arguments.")
elseif(nargin > 4)
    error("Invalid input: Supplied too many arguments.")
else
    if(nargin == 3)
        disp("No initial starting point supplied. Searching for a feasible " + ...
            "point now.")
        [x,~] = l1feas_incl_solver(A,b,1e-08,false);
    end
end

% Check if the initial point is admissible.
A_times_x = A*x;
residual = b - A_times_x;
if(any(abs(residual) > 1e-08))
    warning("The problem has no feasible solutions.")
    x = [];
    return;
end

%% Main algorithm
% Compute the LSQ solution q0 = (A*A.')^{-1}*A*(v - x)
term = A*v - b;
q = (A*A.')\term;

% Compute the descent direction
d = -(A.'*q + (x - v));

% Compute final solution
x = x + d;
end