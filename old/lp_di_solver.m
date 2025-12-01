function [x_lp,v_lp,p_lp,res,count_nnls] = lp_di_solver(A,b,c,p0,tol)
% feas_di_solver   
%                           Computes the solution to the
%                           linear programming problem

%                           max_{p \in \Rm} {<b,p> s.t. A.'*p >= c, p>=0},
%                           and its dual problem up to the tolerance 
%                           level tol, using differential inclusions
%                           and the minimal selection principle.
%   Input
%       A       -   m by n design matrix of the FEAS problem
%       b       -   m dimensional col data vector
%       c       -   n dimensional col data vector
%       p0      -   m dimensional col initial value of the slow system
%       tol     -   small positive number (e.g., 1e-08)
%
%   Output
%       x_lp
%       v_lp
%       p_lp   -   (m,1) array containing the dual solution at
%               hyperparamter t and data b within 
%               tolerance level tol.
%       res
%       count_nnls  -   Number of NNLS calls.

% AUTHORS:
%   The algorithm was designed by Gabriel P. Langlois and Jérôme Darbon.
%   This code was written by Gabriel P. Langlois
%
% REFERENCES:
%   TBC.


%% Initialization
[m,n] = size(A);
count_nnls = 0;
p_lp = p0;
M = [A,eye(m)];
Atop_times_p = A.'*p_lp;

% Check if p0 is an admissible point
if(any((Atop_times_p-c) < -tol) && any(p_lp < -tol))
    error("The point p0 is not a feasible point.")
end

% Compute equicorrelation set and initialize the matrix
eq_set_1 = (abs(Atop_times_p - c) <= tol);
eq_set_2 = (abs(p_lp) <= tol);
eq_set = [eq_set_1;eq_set_2];
K = M(:,eq_set);

% Other quantities
v_lp = zeros(m,1);
w_lp = zeros(m+n,1);
x_lp = zeros(n,1);


while(true)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the NNLS problem 
    % min_{w(eq_set)>=0} ||K*w - b||_{2}^{2}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [w,d] = hinge_lsqnonneg(K,b,tol);
    count_nnls = count_nnls + 1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the maximum admissible descent time over the
    % indices j \in {1,...,n} where abs(<A.'*d,ej>) >= 0.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Timestep for Binf
    Atop_times_d = A.'*d;
    pos_set_1 = Atop_times_d < -tol;
    timestep_1 = inf;
    if(any(pos_set_1))
        term = c-Atop_times_p;
        vec = term./Atop_times_d;
        timestep_1 = min(vec(pos_set_1));
    end

    % Timestep for p>= 0
    pos_set_2 = d < -tol;
    timestep_2 = inf;
    if(any(pos_set_2))
        term5 = p_lp./abs(min(0,d));
        timestep_2 = min(term5(pos_set_2));
    end
    timestep = min(timestep_1,timestep_2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Check for convergence.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(timestep == inf)
        w_lp(eq_set) = w;
        v_lp = w_lp(n+1:m+n);
        x_lp = w_lp(1:n);
        res = norm(d,2);
        break;
    end
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updates all quantities
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    p_lp = p_lp + timestep*d;
    Atop_times_p = Atop_times_p + timestep*Atop_times_d;
    eq_set_1 = (abs(Atop_times_p - c) <= tol);
    eq_set_2 = (abs(p_lp) <= tol);
    eq_set = [eq_set_1;eq_set_2];
    K = M(:,eq_set);
end
end

