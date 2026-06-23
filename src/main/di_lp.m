function [x,p,fval,flag] = di_lp(A,b,c,x,tol,maxit,nnls_solver)
%   DI_LP               Differential inclusions approach to solving
%                       linear programs or certifying unboundedness.
%
%   This function certifies whether the following feasible linear program
%
%   (1) min_{x \in \Rn} \langle\bc,\bx\rangle   subject to A*x \leqslant b
%
%   is bounded and, if so, outputs an optimal solution.
%
%   The differential inclusions approach is a finite-time algorithm that
%   computes the continuous-time solution of the differential inclusions
%   associated to a linear program. The solution consists of finitely
%   many piecewise continuous components calculated from a sequence of
%   nonnegative least-squares (NNLS) problems.
%
%   The main computational cost of this algorithm is the computation of the
%   NNLS problems. The NNLS solver is supplied as a function handle.
%
% -------------------------------------------------------------------------
%   INPUTS
%       A           -   (m x n)-dimensional matrix A
%       b           -   m-dimensional col data vector
%       c           -   n-dimensional col data vector
%       x           -   (Optional) Initial feasible point satisfying
%                       A*x <= b.
%       tol         -   (Optional) Small number specifying the tolerance
%                       (e.g., 1e-08). Default value is derived from the
%                       Frobenius norm of A and the relative size of b.
%       maxit       -   (Optional) Maximum number of iterations in the
%                       outer loop. Default is max(1e6,100*n).
%       nnls_solver -   (Optional) Function handle to an NNLS solver.
%                       Two calling conventions are supported:
%                         [q,d]      = solver(M, rhs, tol, p_warm)
%                         [q,d,veff] = solver(M, rhs, tol, p_warm, v_warm)
%                       The second form is used when the solver accepts a
%                       momentum warm-start vector v_warm. The convention
%                       is detected automatically via nargin(nnls_solver).
%                       Default is @epgd_lsqnonneg.
%
%   OUTPUTS
%       x           -   n-dimensional solution vector to the LP problem.
%                       If unbounded, returns empty field.
%       p           -   m-dimensional dual solution vector. If infeasible,
%                       returns empty field.
%       fval        -   Value of the objective function.
%       flag        -   Returns 0 if x is infeasible or the problem is
%                       unbounded, returns -1 if the outer loop reached
%                       maxit iterations (tolerance issue or exponential
%                       behavior), and returns 1 if successful.
%
% -------------------------------------------------------------------------
%   QUICK EXAMPLE
      % % Generate the data
      % m = 500;
      % n = 1000;
      % x0     = randn(n, 1);
      % A      = randn(m, n);
      % s      = abs(randn(m, 1)) + 1;
      % b      = A * x0 + s;
      % lambda = abs(randn(m, 1)) + 1;  % strictly positive dual variable
      % c      = -A' * lambda;
      % 
      % tol = 1e-08;
      % 
      % % Boundary point via ray casting from x0
      % d  = randn(n, 1);
      % Ad = A * d;
      % idx = Ad > 0;
      % t  = min(s(idx) ./ Ad(idx));  % largest feasible step
      % x1 = x0 + t * d;
      % 
      % [x_linprog,fval_lp] = linprog(c,A,b);
      % [x_di,~,fval_di,~]  = di_lp(A,b,c,x0,tol,[],@epgd_lsqnonneg);
      % 
      % disp(['   <c,x_linprog - x_di> = ', ...
      %     num2str(fval_lp - fval_di), ', achieved with tolerance = ', ...
      %     num2str(tol)])
%
% -------------------------------------------------------------------------


%% Preliminary checks
% Check for an acceptable number of input arguments
if nargin < 3
    error('di_lp: Not enough input arguments.')
end
if ~iscolumn(b)
    error('di_lp: The data vector b is not a column vector.');
end
if ~iscolumn(c)
    error('di_lp: The data vector c is not a column vector.');
end


% Check matrix and vector inputs have appropriate sizes
[m,n] = size(A);
if size(b,1) ~= m
    error(['di_lp: The number of rows of A does not match the ' ...
        'length of the column vector b.'])
end
if size(c,1) ~= n
    error(['di_lp: The number of columns of A does not match the ' ...
        'length of the column vector c.'])
end


% Set defaults before the feasibility phase, which may need them.
if nargin < 5 || isempty(tol)
    tol = min(1e-08, 100*eps*max(norm(A,'fro'), 1)*max(norm(b), 1));
end
if nargin < 6 || isempty(maxit)
    maxit = max(1e6, 100*n);
end
if nargin < 7 || isempty(nnls_solver)
    nnls_solver = @apgd_lsqnonneg;
end
if ~isnumeric(maxit)
    error('di_lp: maxit must be a positive numeric scalar or [].')
end
if ~isa(nnls_solver, 'function_handle')
    error('di_lp: nnls_solver must be a function handle.')
end

% Compute a feasible starting point if one was not supplied.
if nargin < 4 || isempty(x)
    [x, flag_feas] = di_phase1(A, b, tol, maxit, nnls_solver);
    if((flag_feas == 0) || (flag_feas == -1))
        fprintf('The linear program is infeasible within tolerance.')
        x = [];
        flag = 0;
        return;
    end
end


% Check if the supplied initial point is feasible.
fval = c.'*x;
p = zeros(m,1);
flag = 1;
A_times_x = A*x;
residual = b - A_times_x;
if any(-residual > tol)
    fprintf('The initial point is infeasible within tolerance.')
    x = [];
    flag = 0;
    return;
end


%% Preconditioning
% Ruiz equilibration: 10 steps of simultaneous row/column inf-norm scaling.
% Produces d1 (m x 1) and d2 (n x 1) such that diag(d1)*A*diag(d2) has all
% row and column inf-norms near 1. The change of variables x = diag(d2)*x_hat
% transforms the LP to min c_hat'*x_hat s.t. A_hat*x_hat <= b_hat, with
% A_hat = diag(d1)*A*diag(d2), b_hat = d1.*b, c_hat = d2.*c.
% The dual variable transforms as p = diag(d1)*p_hat.
%
% Reference: Ruiz, Daniel. A scaling algorithm to equilibrate both rows and 
% columns norms in matrices. No. RAL-TR-2001-034. CM-P00040415, 2001.
%
d1 = ones(m, 1);
d2 = ones(n, 1);
for k = 1:10
    r   = max(abs(A), [], 2);
    col = max(abs(A), [], 1)';
    r(r == 0)     = 1;
    col(col == 0) = 1;
    sr = sqrt(r);
    sc = sqrt(col);
    A  = (A ./ sr) ./ sc';
    d1 = d1 ./ sr;
    d2 = d2 ./ sc;
end
b         = d1 .* b;
c         = d2 .* c;
x         = x ./ d2;
residual  = d1 .* residual;
A_times_x = d1 .* A_times_x;

%% Algorithm
% Detect solver calling convention once before the loop.
use_v = (nargin(nnls_solver) > 4);

% Initialization
eqset = abs(residual) < tol;
neff = sum(eqset);
v = zeros(m,1);

count = 0;
while true
    count = count + 1;
    if neff > 0
        if use_v
            [q,d,veff] = nnls_solver(A(eqset,:).', -c, tol, ...
                p(eqset), v(eqset));
            v(eqset)  = veff;
            v(~eqset) = 0;
        else
            [q,d] = nnls_solver(A(eqset,:).', -c, tol, p(eqset));
        end
        p(eqset)  = q;
        p(~eqset) = 0;
        d = -d;
    else
        d = -c;
    end

    % Compute the maximal descent time
    A_times_d = A*d;
    tmp = residual ./ A_times_d;
    tmp(eqset | (A_times_d <= 0)) = inf;
    timestep = min(tmp);

    % Check for convergence
    if timestep*tol > 1
        if norm(d,'inf') < tol
            fval = c.'*x;
            x    = d2 .* x;
            p    = d1 .* p;
            return
        else
            disp('Warning: Problem is unbounded.')
            x = [];
            p = [];
            flag = 0;
            fval = [];
            return
        end
    end

    % Update all quantities
    x         = x + timestep*d;
    A_times_x = A_times_x + timestep*A_times_d;
    residual  = b - A_times_x;
    eqset     = abs(residual) < tol;
    neff      = sum(eqset);
    fval      = fval + timestep*(c.'*d);

    if count == maxit
        disp(['Warning: Outer loop did not converge. Either there is ' ...
            'an issue with the tolerance or this problem scales badly ' ...
            'with m and n (possible exponential dependence on m and n). ' ...
            'If you suspect that the tolerance is fine, retry the ' ...
            'algorithm with maxit = inf.'])
        x    = [];
        p    = [];
        flag = -1;
        return
    end
end


end
