function [x,d] = qp_incl_qr_solver(A,b,c,t,tol,x)
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
%   tol     -  Positive scalar. Numerical tolerance, so should
%              small. If none is supplied, will default to tol = 1e-8.
%
%   x       -  (Optional) Starting point. If none is supplied, will default
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
% Check if the number of arguments is correct.
flag_initial_point_supplied = true;
if(nargin < 5 || nargin > 6)
    error("Invalid number of arguments supplied")
elseif(nargin == 5)
    flag_initial_point_supplied = false;
    disp("No initial starting point supplied. " + ...
            "Searching for a feasible point now...")
        [x,Q,R,eqset] = l1feas_incl_qr_solver(A,b,tol);
        R = R.';
end

% Check if the initial point is admissible.
A_times_x = A*x;
residual = b - A_times_x;
if(any(residual < -tol))
    error("The problem has no feasible solutions.")
end


%% Initialize
timestep_tol = max(1e8,1/tol);
if(flag_initial_point_supplied)
    % Compute QR decomposition if initial point was supplied.
    eqset = (abs(residual) <= tol);
    K = A(eqset,:);
    [Q,R] = qr(K.');
else
    K = A(eqset,:);
end
neff = sum(eqset);
opts.UT = true;


%% Main algorithm
while(true)
    % Solve the NNLS problem and compute the descent direction
    if(neff > 0)
        rhs = -(c + t*x);
        [~,d,active_set,Q,R,~] = ...
            nnls_hinge_qr(K.',Q,R,rhs,eqset,opts,tol);
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
        term3 = abs(residual)./term2;
        timestep = min(term3(eqcset));
    end

    % Check for convergence
    if(t>0)
        if(t*timestep > 1 - tol)
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

    % Update the equicorrelation set and assemble the effective matrix
    new_eqset = (abs(residual) <= tol);
    if(neff > 0)
        ind = setxor(find(active_set),find(new_eqset));
        eqset = new_eqset;
        K = A(eqset,:);
        neff = sum(eqset);

        % Update the QR decomposition if one column is added.
        % Else, recompute the QR decomposition from scratch.
        if(isscalar(ind))
            % Extract new element added to eq_set and its column.
            col = A(ind,:).';
            loc = find(find(eqset) == ind);

            % Call MATLAB's [Q,R] = qrinsert(Q,R,loc,col) without overhead
            [~,nr] = size(R);
            R(:,loc+1:nr+1) = R(:,loc:nr);
            R(:,loc) = Q'*col;
            [Q,R] = matlab.internal.math.insertCol(Q,R,loc);
        else
            % Multiple column updates not supported.
            [Q,R] = qr(K.');
        end
    else
        eqset = new_eqset;
        K = A(eqset,:);
        neff = sum(eqset);
        [Q,R] = qr(K.');
    end
end
end