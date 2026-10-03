function [x,p,fval,flag] = di_lp(A,b,c,x,tol,maxit,nnls_solver)
%   DI_LP           Gradient inclusion solver for linear programs.
%
%   Solves, or certifies unbounded,
%
%   (1) min_{x in R^n}  c'*x   subject to  A*x <= b.
%
%   The gradient inclusion trajectory of (1) reaches an optimal vertex in
%   finitely many pieces.  Each piece costs one NNLS solve on the active
%   face; that solve dominates the work and is supplied as a handle.
%
% -------------------------------------------------------------------------
%   INPUTS
%       A             -   (m x n) constraint matrix.
%       b             -   m-dimensional column vector.
%       c             -   n-dimensional cost column vector.
%       x             -   (Optional) Feasible start, A*x <= b.  Default is
%                         a point from di_phase1.
%       tol           -   (Optional) Tolerance.  Default is
%                         min(1e-8, 100*eps*||A||_F*||b||).
%       maxit         -   (Optional) Outer iteration cap.  Default is
%                         max(1e6, 100*n).
%       nnls_solver   -   (Optional) Handle to an NNLS solver, called as
%                         [q,d] = solver(M,rhs,tol,p_warm), or as
%                         [q,d,v] = solver(M,rhs,tol,p_warm,v_warm) when it
%                         warm-starts the power iteration for ||M||_2;
%                         nargin picks the form.  Default is @apgd_lsqnonneg.
%
%   OUTPUTS
%       x             -   Optimal solution; empty if infeasible or
%                         unbounded.
%       p             -   m-dimensional dual; empty if infeasible.
%       fval          -   Objective value at x.
%       flag          -   1 converged, 0 infeasible or unbounded,
%                         -1 reached maxit.
%
% -------------------------------------------------------------------------
%   QUICK EXAMPLE
      % rng(1);
      % m = 500;  n = 1000;
      % x0 = randn(n,1);
      % A  = randn(m,n);
      % b  = A*x0 + abs(randn(m,1)) + 1;
      % c  = -A' * (abs(randn(m,1)) + 1);      % bounded below
      %
      % [~, fval_lp]        = linprog(c, A, b);
      % [~, ~, fval_di, ~]  = di_lp(A, b, c, x0, 1e-8);
      % fprintf('fval gap: %e\n', abs(fval_lp - fval_di))
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
        fprintf('The linear program is infeasible within tolerance.\n')
        x    = [];
        p    = [];
        fval = [];
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
    fprintf('The initial point is infeasible within tolerance.\n')
    x    = [];
    p    = [];
    fval = [];
    flag = 0;
    return;
end


%% Preconditioning
% Ruiz equilibration: 10 sweeps of row/column inf-norm scaling, giving d1
% and d2 with diag(d1)*A*diag(d2) of inf-norms near 1.  Under x = diag(d2)*xh
% the LP becomes min (d2.*c)'*xh s.t. diag(d1)*A*diag(d2)*xh <= d1.*b, and
% the dual maps back as p = diag(d1)*ph.
%
% Reference: D. Ruiz, A scaling algorithm to equilibrate both rows and
% columns norms in matrices, RAL-TR-2001-034, 2001.
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
