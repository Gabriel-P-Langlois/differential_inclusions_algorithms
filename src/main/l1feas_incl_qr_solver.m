function [x,Q,R,eqset] = l1feas_incl_qr_solver(A,b,tol)
% L1FEAS_INCL_QR_SOLVER     This function determines if there exists
%                           a feasible point of the linear system Ax<=b
%                           by solving the inequality form of the basis
%                           pursuit problem:
%
%                           min_{x \in \Rn} ||x||_{1} s.t. Ax<=b.
%
%                           This is achieved by solving its dual problem
%
%                           min_{p \in \Rm} <b,p> 
%                                s.t. ||(-Atop*p)||_{\inf} <= 1 and p>=0.
%
%                           Note that p = 0 is always feasible point of the
%                           dual problem, and there are other alternatives
%                           as well.
%
%                           This function uses QR updates to accelerate
%                           the computations.
% INPUT:
%   A       - (m x n)-dimensional vector.
%   b       - m-dimensional vector.
%   tol         -   (Optional) Small number specifying the tolerance. 
%                   If unsupplied, set tol = 1e-08;
%
% OUTPUT:
%   x       - n-dimensional vector satisfying A*x <= b, if one exists.
%             If no vectors exist, returns an empty field instead.
%   Q       - (m times m) matrix Q factorization of A at the end.
%   R       - (m times n) matrix R factorization of A at the end.
%
% Notes:
%   1) Code is slowed down by the formation of the matrix [A,eye(m)].


%% Initialization
% Options, placeholders and initial conditions.
timestep_tol = max(1e8,1/tol);
[m,n] = size(A);
tol_minus = 1-tol;
M = [A,eye(m)];
x = zeros(n,1);

% Choice of the initial condition
p = zeros(m,1);

w_di = zeros(m+n,1);

Atop_times_p = A.'*p;
eqset_1 = (abs(Atop_times_p) >= tol_minus);
eqset_2 = (abs(p) <= tol);
eqset = [eqset_1;eqset_2];

vec_of_signs = [sign(-Atop_times_p);ones(m,1)];
vec_of_signs(vec_of_signs == 0) = 1;

K = M(:,eqset).*(vec_of_signs(eqset).');
neff = sum(eqset);

% Perform the initial QR decomposition and set the opts field.
% For the special choice of p = zeros(m,1), Q=R=eye(m).
Q = eye(m); R = eye(m); 
opts.UT = true;

while(true)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the NNLS problem 
    % min_{w(eq_set)>=0} ||K*w - b||_{2}^{2}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(neff > 0)
        [w,d,active_set,Q,R,~] = nnls_hinge_qr(K,Q,R,b,eqset,opts,tol);
    else
        d = -b;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the maximum admissible descent time over the
    % indices j \in {1,...,n} where abs(<A.'*d,ej>) >= 0.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Timestep for Binf
    Atop_times_d = A.'*d;
    pos_set_1 = abs(Atop_times_d) > tol;
    timestep_1 = inf;
    if(any(pos_set_1))
        term1 = vec_of_signs(1:n).*Atop_times_d;
        term2 = vec_of_signs(1:n).*Atop_times_p;
        term3 = sign(term1);
        term4 = term3 - term2;
        vec = term4./term1;
        timestep_1 = min(vec(pos_set_1));
    end

    % Timestep for p>= 0
    pos_set_2 = (d < -tol);
    timestep_2 = inf;
    if(any(pos_set_2))
        term5 = p./abs(min(0,d));
        timestep_2 = min(term5(pos_set_2));
    end
    timestep = min(timestep_1,timestep_2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Check for convergence.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(timestep > timestep_tol)
        w_di(eqset) = vec_of_signs(eqset).*w;
        x = w_di(1:n);
        if(norm(d,'inf') > tol)
            warning("No feasible points have been found.")
        end

        % Recover the QR decomposition of A(:,eqset)
        R = R.*vec_of_signs(eqset).';
        eqset(n+1:end) = [];
        R(:, sum(eqset)+1:end) = [];
        break;
    end
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updates all quantities
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    p = p + timestep*d;
    Atop_times_p = Atop_times_p + timestep*Atop_times_d;
    vec_of_signs = [sign(-Atop_times_p);ones(m,1)];
    vec_of_signs(vec_of_signs == 0) = 1;
    
    % Update the equicorrelation set and assemble the effective matrix
    new_eqset_1 = (abs(Atop_times_p) >= tol_minus);
    new_eqset_2 = (abs(p) <= tol);
    new_eqset = [new_eqset_1;new_eqset_2];

    if(neff > 0)
        ind = setxor(find(active_set),find(new_eqset));
        eqset = new_eqset;
        K = M(:,eqset).*(vec_of_signs(eqset).');
        neff = sum(eqset);
    
        % Update the QR decomposition if one column is added.
        % Else, recompute the QR decomposition from scratch.
        if(isscalar(ind))
            % Extract new element added to eq_set and its column.
            col = M(:,ind);
            loc = find(find(eqset) == ind);

            % Call MATLAB's [Q,R] = qrinsert(Q,R,loc,col) without overhead
            [~,nr] = size(R);
            R(:,loc+1:nr+1) = R(:,loc:nr);
            R(:,loc) = Q'*col;
            [Q,R] = matlab.internal.math.insertCol(Q,R,loc);
        else
            % Multiple column updates not supported.
            [Q,R] = qr(K);
        end
    else
        eqset = new_eqset;
        K = M(:,eqset).*(vec_of_signs(eqset).');
        neff = sum(eqset);
        [Q,R] = qr(K);
    end
    
end
end