function [x,d] = qp_incl_solver(A,b,c,t,tol,use_nnls_di,x)
% QP_INCL_SOLVER    This function computes an exact solution of the
%                   constrained quadratic problem:
%
%                   min_{x \in \Rn} 0.5*t*||x||_{2}^{2} + <c,x>,
%                                   subject to Ax<=b,
%
%                   by integrating the differential inclusions
%                   arising from this problem.
%
% INPUT:
%   A       -  (m x n)-dimensional matrix.
%   b       -  m-dimensional vector
%   c       -  n-dimensional vector
%   t       -  nonnegative parameter.
%              for which a feasible point can be computed via its dual.
%   tol     -  (Optional) positive scalar. Numerical tolerance, so should
%              be small. If none is supplied, will default to tol = 1e-8.
%
%   use_nnls_di - (Optional) Boolean field specifying whether to use the
%                 differential inclusions approach for solving the NNLS
%                 problem. If not specified, set to false, in which case
%                 the method of hinges is used.
%   x       -  (Optional) starting point. If none is supplied, will default
%              to solving the Phase I problem
%              min_{x \in \Rn} ||x||_{1} subject to Ax <= b,
%
%
% OUTPUT:
%   x       - Optimal solution to the problem, if it exists. If not,
%             return empty.
%   d       - Final descent direction before convergence. If t = 0,
%             then an optimal solution exists iff d = 0.


%% Preliminary checks
% Check if the arguments are correct
flag_initial_point_supplied = true;
if(nargin < 4)
    error("Invalid input: Supplied too few arguments.")
elseif(nargin > 7)
    error("Invalid input: Supplied too many arguments.")
else
    if(nargin < 7)
        flag_initial_point_supplied = false;
        if(nargin == 4)
            tol = 1e-8;
            use_nnls_di = false;
        elseif(nargin == 5)
            use_nnls_di = false;
        end
        disp("No initial starting point supplied. " + ...
            "Searching for a feasible point now.")
        [x,~] = l1feas_incl_solver(A,b,tol,use_nnls_di);
    end
end

% Check if the initial point is admissible.
A_times_x = A*x;
residual = b - A_times_x;
if(any(residual < -tol))
    if(flag_initial_point_supplied)
        error("Supplied starting point is infeasible.")
    else
        warning("The problem has no feasible solutions.")
        x = [];
        d = [];
        return;
    end
end


%% Initialize
timestep_tol = max(1e8,1/tol);
eqset = (abs(residual) <= tol);
K = A(eqset,:);
neff = sum(eqset);


%% Main algorithm
while(true)
    % Solve the NNLS problem and compute the descent direction
    if(neff > 0)
        ceff = -(c + t*x);
        if(use_nnls_di)
            [~,d,~] = nnls_inclusions(K.',ceff,zeros(neff,1),tol);
        else
            [~,d] = nnls_hinge(K.',ceff,tol,true(neff,1));
        end
        d = -d;
    else
        d = -(c + t*x);
    end

    % Compute the maximal descent time
    eqcset = (abs(residual) > tol);
    timestep = inf;
    if(any(eqcset))
        term1 = A*d;
        term2 = max(0,term1);
        term3 = residual./term2;
        timestep = min(term3(eqcset));
    end

    % Check for convergence
    if(t>0)
        if(t*timestep >= 1 - tol)
            x = x + d/t;
            break;
        end
    elseif(timestep > timestep_tol)
        if(norm(d,'inf') < tol)
            break;
        else
            x = [];
            warning("Problem is unbounded.");
            break;
        end
    end

    % Update all quantities.
    x = x + timestep*d;
    A_times_x = A_times_x + timestep*term1;
    residual = b - A_times_x;
    eqset = (abs(residual) <= tol);

    K = A(eqset,:);
    neff = sum(eqset);
end
end