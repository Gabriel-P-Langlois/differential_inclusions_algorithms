function [x,p,fval,flag] = di_qp(A,b,c,Q,x,tol,maxit,nnls_solver,tol_nnls)
%   DI_QP           Gradient inclusion solver for convex quadratic programs.
%
%   Approximately solves, with Q symmetric positive definite,
%
%   (1) min_{x in R^n}  (1/2)*x'*Q*x + c'*x   subject to  A*x <= b.
%
%   On each piece the trajectory follows d = -(Q*x + c + A_eq'*p), where
%   A_eq = A(eqset,:) is the active block and p >= 0 solves the NNLS
%   subproblem min_{p >= 0} (1/2)*||A_eq'*p + Q*x + c||^2.  The step is the
%   smaller of the time to hit a new constraint and the exact line search
%   time ||d||^2/(d'*Q*d).  Each iteration costs one product Q*d and one
%   NNLS solve.
%
%   Unlike the LP case and the case Q = t*I, the method need not stop in
%   finite time; it descends to an approximate KKT point.
%
%   This is Algorithm 2 of the paper for SPD Q only.  For SPSD Q, Algorithm
%   2 also stops when the step is infinite and returns a direction of
%   unboundedness; that exit is not implemented here.
%
% -------------------------------------------------------------------------
%   INPUTS
%       A             -   (m x n) constraint matrix.
%       b             -   m-dimensional column vector.
%       c             -   n-dimensional cost column vector.
%       Q             -   (n x n) symmetric positive definite matrix, or a
%                         handle v -> Q*v.
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
%       tol_nnls      -   (Optional) RELATIVE accuracy of the inner NNLS
%                         solve, decoupled from tol.  The right-hand side
%                         -grad is normalized before each solve, so the
%                         tolerance is scale invariant.  Keep it tight: a
%                         loose solve returns a direction that violates
%                         A_eq*d <= 0 and the active set then chatters.
%                         Default is 1e-10.  Membership tests and the
%                         ||d|| < tol stop keep tol.
%
%   OUTPUTS
%       x             -   Approximate solution; empty if infeasible.
%       p             -   m-dimensional dual.
%       fval          -   Objective value at x.
%       flag          -   1 converged, 0 infeasible, -1 reached maxit.
%
% -------------------------------------------------------------------------
%   QUICK EXAMPLE
      % rng(1);
      % m = 200;  n = 100;
      % A  = randn(m,n);
      % x0 = randn(n,1);
      % b  = A*x0 + abs(randn(m,1)) + 1;
      % B  = randn(n);  Q = B'*B + eye(n);  c = randn(n,1);
      %
      % opts = optimoptions('quadprog','Display','off');
      % [~, fval_qp]       = quadprog(Q, c, A, b, [],[],[],[],[], opts);
      % [~, ~, fval_di, ~] = di_qp(A, b, c, Q, x0, 1e-8);
      % fprintf('fval gap: %e\n', abs(fval_qp - fval_di))
%
% -------------------------------------------------------------------------


%% Preliminary checks
if nargin < 4
    error('di_qp: Not enough input arguments.')
end
if ~iscolumn(b)
    error('di_qp: The data vector b is not a column vector.');
end
if ~iscolumn(c)
    error('di_qp: The data vector c is not a column vector.');
end

[m,n] = size(A);
if size(b,1) ~= m
    error(['di_qp: The number of rows of A does not match the ' ...
        'length of the column vector b.'])
end
if size(c,1) ~= n
    error(['di_qp: The number of columns of A does not match the ' ...
        'length of the column vector c.'])
end
if ~isa(Q, 'function_handle') && ~isequal(size(Q), [n, n])
    error('di_qp: Q must be an (n x n) matrix or a function handle v -> Q*v.')
end

% Wrap Q into a uniform function handle.
if isa(Q, 'function_handle')
    Qfun = Q;
else
    Qfun = @(v) Q*v;
end


% Set defaults.
if nargin < 6 || isempty(tol)
    tol = min(1e-08, 100*eps*max(norm(A,'fro'), 1)*max(norm(b), 1));
end
if nargin < 7 || isempty(maxit)
    maxit = max(1e6, 100*n);
end
if nargin < 8 || isempty(nnls_solver)
    nnls_solver = @apgd_lsqnonneg;
end
if nargin < 9 || isempty(tol_nnls)
    tol_nnls = 1e-10;     % relative NNLS accuracy (RHS normalized to unit norm)
end
if ~isnumeric(maxit)
    error('di_qp: maxit must be a positive numeric scalar or [].')
end
if ~isa(nnls_solver, 'function_handle')
    error('di_qp: nnls_solver must be a function handle.')
end

% Compute a feasible starting point if one was not supplied.
if nargin < 5 || isempty(x)
    [x, flag_feas] = di_phase1(A, b, tol, maxit, nnls_solver);
    if (flag_feas == 0) || (flag_feas == -1)
        fprintf('The quadratic program is infeasible within tolerance.\n')
        x    = [];
        p    = [];
        fval = [];
        flag = 0;
        return;
    end
end

% Check that the supplied (or computed) initial point is feasible.
p = zeros(m,1);
flag = 1;
A_times_x = A*x;
residual   = b - A_times_x;
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
% the QP keeps its form with A_hat = diag(d1)*A*diag(d2), b_hat = d1.*b,
% c_hat = d2.*c, Q_hat = diag(d2)*Q*diag(d2), and the dual maps back as
% p = diag(d1)*ph.
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
Qfun      = @(v) d2 .* Qfun(d2 .* v);

%% Algorithm
% Detect solver calling convention once before the loop.
use_v = (nargin(nnls_solver) > 4);

% Initialization
eqset = abs(residual) < tol;
neff  = sum(eqset);
v     = zeros(m,1);

% Objective gradient, then maintained incrementally: x advances by
% timestep*d, so grad <- grad + timestep*(Q*d) with Q*d already formed for
% the line search.  This halves the Q applications, the dominant cost.
grad = Qfun(x) + c;

count = 0;
while true
    count = count + 1;

    % Solve the NNLS subproblem on the active face to obtain the dual
    % variable q and the instantaneous descent direction d.
    if neff > 0
        % Normalize the right-hand side so tol_nnls is scale invariant: the
        % multiplier scale tracks ||grad||, which shrinks as a mesh refines.
        % NNLS is positively homogeneous, so q and d rescale by s.
        s   = max(norm(grad), realmin);
        rhs = -grad / s;
        if use_v
            [q,d,veff] = nnls_solver(A(eqset,:).', rhs, tol_nnls, ...
                p(eqset)/s, v(eqset));
            v(eqset)  = veff;
            v(~eqset) = 0;
        else
            [q,d] = nnls_solver(A(eqset,:).', rhs, tol_nnls, p(eqset)/s);
        end
        q = s * q;
        d = s * d;
        p(eqset)  = q;
        p(~eqset) = 0;
        d = -d;
    else
        d = -grad;
    end

    % Check for convergence: d is the projected negative gradient, so
    % ||d|| < tol is the KKT stationarity condition up to tolerance.
    if norm(d) < tol
        fval = 0.5*(x'*Qfun(x)) + c'*x;
        x    = d2 .* x;
        p    = d1 .* p;
        return
    end

    % Compute the time to hit the nearest inactive constraint.
    A_times_d = A*d;

    % NNLS optimality requires A(eqset,:)*d <= 0; finite tolerance can flip
    % individual rows.  Repair by projecting d off each offending row.
    bad = eqset & (A_times_d > 0);
    if any(bad)
        A_bad = A(bad,:);
        d = d - A_bad' * (A_times_d(bad) ./ sum(A_bad.^2, 2));
        A_times_d = A*d;
    end
    tmp = residual ./ A_times_d;
    tmp(eqset | (A_times_d <= 0)) = inf;
    t_constr = min(tmp);

    % Compute the exact quadratic line search time along d.  Cache Q*d: it
    % also advances the gradient after the step (grad <- grad + timestep*Q*d).
    Qd     = Qfun(d);
    t_quad = (d'*d) / (d'*Qd);

    % Take the minimum: move as far as possible without leaving feasibility,
    % but no further than the unconstrained minimizer along d.
    timestep = min(t_constr, t_quad);

    % Update all quantities.
    x         = x + timestep*d;
    grad      = grad + timestep*Qd;
    A_times_x = A_times_x + timestep*A_times_d;
    residual  = b - A_times_x;
    eqset     = abs(residual) < tol;
    neff      = sum(eqset);

    if count == maxit
        disp(['Warning: Outer loop did not converge. The tolerance may ' ...
            'be too tight or the problem may be ill-conditioned. ' ...
            'Retry with a looser tolerance or a larger maxit.'])
        x    = [];
        p    = [];
        fval = [];
        flag = -1;
        return
    end
end


end
