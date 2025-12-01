function [x_g, y_g, p_g, v_g, b_g, w_g, d_g,count_lsq] = ...
    feas_di_greedy(A,b,tol)
% feas_di_greedy  
%                           Computes a feasible point to the
%                           FEAS primal problem
%                           min_{x \in \Rn} {0.5\normsq{max(0,Ax-b)}/t 
%                                           + ||x||_1},
%                           using the greedy differential inclusions
%                           algorithm and the minimal selection principle.
%
%                           This code uses the QR decomposition and column 
%                           updates to speed up the calculations.
%   Input
%       A       -   m by n design matrix of the FEAS problem
%       b       -   m dimensional col data vector of the FEAS problem
%       tol     -   small positive number (e.g., 1e-08)
%
%   Output
%       x_g   -   (n,1) array containing the primal solution at 
%                   hyperparameter t and data b within 
%                   tolerance level tol.
%
%       y_g
%
%       p_g   -   (m,1) array containing the dual solution at
%                   hyperparamter t and data b within 
%                   tolerance level tol.
%
%       v_g
%
%       b_g
%
%       w_g
%
%       d_g
%
%       count_lsq  -   Number of NNLS calls.

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

x_g = zeros(m+n,1);
y_g = zeros(m+n,1);
p_g = max(0,-b);
b_g = b;

t0 = max(norm(A.'*p_g,'inf'),max(p_g));
p_g = p_g/t0;
t = t0;

count_lsq = 0;

Atop_times_p = A.'*p_g;
eq_set_1 = (abs(Atop_times_p) >= tol_minus);
eq_set_2 = (abs(p_g) <= tol);
eq_set = [eq_set_1;eq_set_2];

vec_of_signs = [sign(-Atop_times_p);ones(m,1)];
K = M(:,eq_set).*(vec_of_signs(eq_set).');

while(true)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the NNLS problem 
    % min_{w(eq_set)>=0} ||K*w - (b + t*sol_p)||_{2}^{2}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rhs = -t*p_g;
    z = K\rhs;
    xi = K*z - rhs;
    count_lsq = count_lsq + 1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the maximum admissible descent time over the
    % indices j \in {1,...,n} where abs(<A.'*d,ej>) >= 0.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Timestep for Binf
    Atop_times_xi = A.'*xi;
    pos_set_1 = abs(Atop_times_xi) > tol;
    Ck_1 = inf;
    if(any(pos_set_1))
        term1 = vec_of_signs(1:n).*Atop_times_xi;
        term2 = vec_of_signs(1:n).*Atop_times_p;
        term3 = sign(term1);
        term4 = term3 - term2;
        vec = term4./term1;
        Ck_1 = min(vec(pos_set_1));
    end

    % Timestep for p>= 0
    pos_set_2 = xi < -tol;
    Ck_2 = inf;
    if(any(pos_set_2))
        term5 = p_g./abs(min(0,xi));
        Ck_2 = min(term5(pos_set_2));
    end
    Ck = min(Ck_1,Ck_2);
    fac = t/(1+t*Ck);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Readjust data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    tmp = vec_of_signs.*x_g;
    w_g = zeros(m+n,1); w_g(eq_set) = max(0,-tmp(eq_set)-z);
    z = z + w_g(eq_set);
    q = K*(-w_g(eq_set)/t);

    x_g(eq_set) = x_g(eq_set) + (1-fac/t)*(vec_of_signs(eq_set).*z);
    y_g = y_g + (1-fac/t)*(vec_of_signs.*w_g);
    b_g = b_g + (fac-t)*q;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Check for convergence.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(Ck == inf)
        break;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updates dual solution, timestep and eq sets.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    timestep = (1/fac - 1/t);
    t = fac;
    p_g = p_g + timestep*xi;
    Atop_times_p = Atop_times_p + timestep*Atop_times_xi;
    eq_set_1 = (abs(Atop_times_p) >= tol_minus);
    eq_set_2 = (abs(p_g) <= tol);
    eq_set = [eq_set_1;eq_set_2];
    vec_of_signs = [sign(-Atop_times_p);ones(m,1)];
    K = M(:,eq_set).*(vec_of_signs(eq_set).');


    % DEBUG
    % disp([norm(A.'*xi,'inf'),xi.'*b])
end

% Configure the output
d_g = M*(vec_of_signs.*x_g) - b_g;
v_g = x_g(n+1:n+m);
x_g(n+1:n+m) = [];
end