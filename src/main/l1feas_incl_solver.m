function x = l1feas_incl_solver(A,b,tol,use_nnls_di)
% L1FEAS_INCL_SOLVER        This function determines if there exists
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
% INPUT:
%   A       - (m x n)-dimensional vector.
%   b       - m-dimensional vector.
%   tol         -   (Optional) Small number specifying the tolerance. 
%                   If unsupplied, set tol = 1e-08;
%   use_nnls_di - (Optional) Boolean field specifying whether to use the
%                 differential inclusions approach for solving the NNLS
%                 problem. If not specified, set to false, in which case
%                 the method of hinges is used.
%
% OUTPUT:
%   x       - n-dimensional vector satisfying A*x <= b, if one exists.
%             If no vectors exist, returns an empty field instead.
%   p       - m-dimensional vector feasible solution to the dual problem.
%             If the problem is unbounded, return an empty field instead.


%% Initialization
% Options, placeholders and initial conditions.
timestep_tol = max(1e8,1/tol);
[m,n] = size(A);
tol_minus = 1-tol;
M = [A,eye(m)];
x = zeros(n,1);
p = zeros(m,1);

w_di = zeros(m+n,1);

%v_di = zeros(m,1);
%d_di = zeros(m,1);

count_nnls = 0;

Atop_times_p = A.'*p;
eqset_1 = (abs(Atop_times_p) >= tol_minus);
eqset_2 = (abs(p) <= tol);
eqset = [eqset_1;eqset_2];

vec_of_signs = [sign(-Atop_times_p);ones(m,1)];
vec_of_signs(vec_of_signs == 0) = 1;

K = M(:,eqset).*(vec_of_signs(eqset).');
neff = sum(eqset);

k = 0;
while(true)
    k = k + 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the NNLS problem 
    % min_{w(eq_set)>=0} ||K*w - b||_{2}^{2}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(neff > 0)
        if(use_nnls_di)
            [w,d,~] = nnls_inclusions(K,b,zeros(neff,1),1e-08);
        else
            [w,d] = nnls_hinge(K,b,tol,true(neff,1));
        end
    else
        d = -b;
    end
    count_nnls = count_nnls + 1;
    
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
        % v_di = w_di(n+1:m+n);
        % d_di = M*w_di - b;
        if(norm(d,'inf') > tol)
            warning("No feasible points have been found.")
        end
        break;
    end
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updates all quantities
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    p = p + timestep*d;
    Atop_times_p = Atop_times_p + timestep*Atop_times_d;

    eqset_1 = (abs(Atop_times_p) >= tol_minus);
    eqset_2 = (abs(p) <= tol);
    eqset = [eqset_1;eqset_2];

    vec_of_signs = [sign(-Atop_times_p);ones(m,1)];
    vec_of_signs(vec_of_signs == 0) = 1;

    K = M(:,eqset).*(vec_of_signs(eqset).');
    neff = sum(eqset);
end
end