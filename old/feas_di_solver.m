function [x_di,p_di,v_di,d_di,count_nnls] = feas_di_solver(A,b,t,p0,tol)
% feas_di_solver   
%                           Computes the primal and dual solutions to the
%                           FEAS primal problem
%                           min_{x \in \Rn} {0.5\normsq{max(0,Ax-b)}/t 
%                                           + ||x||_1},
%                           and its dual problem up to the tolerance 
%                           level tol, using differential inclusions
%                           and the minimal selection principle.
%
%                           This code computes the solution to the problem 
%                           above along a regularization path, starting 
%                           from t(1) until t(length(t)). In particular, 
%                           it reuses a previous computed dual solution
%                           sol_p(k) as the initial starting point of 
%                           computation of the next iterate k + 1.
%
%                           This code uses the QR decomposition and column 
%                           updates to speed up the calculations.
%   Input
%       A       -   m by n design matrix of the FEAS problem
%       b       -   m dimensional col data vector of the FEAS problem
%       p0      -   m dimensional col initial value of the slow system
%       t       -   nonnegative hyperparameters of the FEAS problem
%       tol     -   small positive number (e.g., 1e-08)
%
%   Output
%       x_di   -   (n,1) array containing the primal solution at 
%                   hyperparameter t and data b within 
%                   tolerance level tol.
%       p_di   -   (m,1) array containing the dual solution at
%                   hyperparamter t and data b within 
%                   tolerance level tol.
%       v_di   -    (m,1) array containing the remaining coefficient
%                   of the vector w satisfying d_di = M*w - (t*p_di + b);
%
%       d    -      (m,1) array residual vector d_di = M*w - (t*p_di + b)
%       count_nnls  -   Number of NNLS calls.

% AUTHORS:
%   The algorithm was designed by Gabriel P. Langlois and Jérôme Darbon.
%   This code was written by Gabriel P. Langlois
%
% REFERENCES:
%   TBC.


%% Initialization
% Options, placeholders and initial conditions.
[m,n] = size(A);
tol_minus = 1-tol;
M = [A,eye(m)];
x_di = zeros(n,1);
v_di = zeros(m,1);
w_di = zeros(m+n,1);
d_di = zeros(m,1);
p_di = p0;
count_nnls = 0;

Atop_times_p = A.'*p_di;
eq_set_1 = (abs(Atop_times_p) >= tol_minus);
eq_set_2 = (abs(p_di) <= tol);
eq_set = [eq_set_1;eq_set_2];

vec_of_signs = [sign(-Atop_times_p);ones(m,1)];
K = M(:,eq_set).*(vec_of_signs(eq_set).');

while(true)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the NNLS problem 
    % min_{w(eq_set)>=0} ||K*w - (b + t*sol_p)||_{2}^{2}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rhs = b + t*p_di;
    [w,d] = hinge_lsqnonneg(K,rhs,tol);
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
    pos_set_2 = d < -tol;
    timestep_2 = inf;
    if(any(pos_set_2))
        term5 = p_di./abs(min(0,d));
        timestep_2 = min(term5(pos_set_2));
    end
    timestep = min(timestep_1,timestep_2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Check for convergence.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(timestep == inf)
        w_di(eq_set) = vec_of_signs(eq_set).*w;
        x_di = w_di(1:n);
        v_di = w_di(n+1:m+n);
        d_di = M*w_di - b - t*p_di;
        if(t > 0)
            p_di = p_di + d/t;
        end
        break;
    elseif(t>0 && timestep*t > tol_minus)
        w_di(eq_set) = vec_of_signs(eq_set).*w;
        x_di = w_di(1:n);
        v_di = w_di(n+1:m+n);
        p_di = p_di + d/t;
        d_di = M*w_di - b - t*p_di;
        break;
    end
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updates all quantities
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    p_di = p_di + timestep*d;
    Atop_times_p = Atop_times_p + timestep*Atop_times_d;
    eq_set_1 = (abs(Atop_times_p) >= tol_minus);
    eq_set_2 = (abs(p_di) <= tol);
    eq_set = [eq_set_1;eq_set_2];

    vec_of_signs = [sign(-Atop_times_p);ones(m,1)];
    K = M(:,eq_set).*(vec_of_signs(eq_set).');
end
end

