function [x,p,fval,flag] = di_qp(A,b,c,Q,x,tol,maxit,nnls_solver,tol_nnls)
%   DI_QP               Differential inclusions approach to approximately
%                       solving convex quadratic programs.
%
%   This function computes an approximate optimal solution of the following
%   feasible convex quadratic program with a symmetric positive definite Q:
%
%   (1) min_{x \in \Rn} (1/2)x'*Q*x + c'*x   subject to A*x <= b
%
%   The differential inclusions approach computes the continuous-time
%   solution of the differential inclusions associated to problem (1). On
%   each local interval, the solution moves along the instantaneous descent
%   direction d = -(Q*x + c + A_eq'*p), where A_eq = A(eqset,:) is the
%   active inequality block and p >= 0 solves the NNLS subproblem
%
%       min_{p >= 0}  (1/2)*||A_eq'*p + Q*x + c||^2.
%
%   The step size is the minimum of the time to hit a new constraint and
%   the exact quadratic line search time ||d||^2 / (d'*Q*d).
%
%   Unlike the LP case and the case Q = t*I, the algorithm does not
%   terminate in finite time in general; it is a descent algorithm that
%   converges to an approximate KKT point.
%
%   The main computational cost per iteration is a matrix-vector product
%   Q*x and a NNLS subproblem on the active face.
%
% -------------------------------------------------------------------------
%   INPUTS
%       A           -   (m x n)-dimensional matrix A
%       b           -   m-dimensional col data vector
%       c           -   n-dimensional col data vector
%       Q           -   Either an (n x n) symmetric positive definite matrix
%                       or a function handle of the form v -> Q*v.
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
%       tol_nnls    -   (Optional) RELATIVE accuracy for the inner NNLS
%                       subproblem, decoupled from the outer tolerance tol.
%                       The NNLS right-hand side -grad is normalized to unit
%                       norm before each solve, so tol_nnls is a scale-invariant
%                       relative tolerance: the multiplier scale tracks ||grad||,
%                       which varies by orders of magnitude across problems and
%                       shrinks as a PDE mesh refines.  The inner solve must be
%                       tight -- a loose NNLS returns a descent direction that
%                       violates the active constraints' optimality
%                       A(eqset,:)*d <= 0, making the outer active-set path
%                       chatter (see the "bad"-row repair below).  Default is
%                       1e-10.  Only the inner solve uses tol_nnls; the
%                       active-set membership test and the outer convergence
%                       test ||d|| < tol keep tol.
%
%   OUTPUTS
%       x           -   n-dimensional approximate solution to problem (1).
%                       Empty if infeasible.
%       p           -   m-dimensional dual solution vector.
%       fval        -   Value of the objective function at x.
%       flag        -   Returns 0 if x is infeasible, returns -1 if the
%                       outer loop reached maxit without converging, and
%                       returns 1 if successful.
%
% -------------------------------------------------------------------------
%   QUICK EXAMPLE
      % % Generate the data
      % rng(1);
      % m = 200;
      % n = 100;
      % A      = randn(m, n);
      % x0     = randn(n, 1);
      % s      = abs(randn(m, 1)) + 1;
      % b      = A * x0 + s;
      % B      = randn(n, n);
      % Q      = B' * B + eye(n);    % random SPD matrix
      % c      = randn(n, 1);
      %
      % tol    = 1e-08;
      % opts   = optimoptions('quadprog', 'Display', 'off');
      % [x_qp, fval_qp] = quadprog(Q, c, A, b, [], [], [], [], [], opts);
      % [x_di, ~, fval_di, ~] = di_qp(A, b, c, Q, x0, tol);
      %
      % disp(['   fval_quadprog = ', num2str(fval_qp)])
      % disp(['   fval_di_qp    = ', num2str(fval_di)])
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
% Ruiz equilibration: 10 steps of simultaneous row/column inf-norm scaling.
% Produces d1 (m x 1) and d2 (n x 1) such that diag(d1)*A*diag(d2) has all
% row and column inf-norms near 1. The change of variables x = diag(d2)*x_hat
% transforms the QP to min (1/2)*x_hat'*Q_hat*x_hat + c_hat'*x_hat s.t.
% A_hat*x_hat <= b_hat, with A_hat = diag(d1)*A*diag(d2), b_hat = d1.*b,
% c_hat = d2.*c, Q_hat = diag(d2)*Q*diag(d2).
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
Qfun      = @(v) d2 .* Qfun(d2 .* v);

%% Algorithm
% Detect solver calling convention once before the loop.
use_v = (nargin(nnls_solver) > 4);

% Initialization
eqset = abs(residual) < tol;
neff  = sum(eqset);
v     = zeros(m,1);

% Gradient of the objective at the current iterate.  Maintained incrementally
% inside the loop: Q is linear and x advances by timestep*d, so
% grad <- grad + timestep*(Q*d), and Q*d is already formed for the line search.
% This removes the standalone Q*x evaluation, roughly halving the number of Q
% applications (the dominant cost) per iteration.
grad = Qfun(x) + c;

count = 0;
while true
    count = count + 1;

    % Solve the NNLS subproblem on the active face to obtain the dual
    % variable q and the instantaneous descent direction d.
    if neff > 0
        % Normalize the NNLS right-hand side to unit norm so tol_nnls is a
        % relative, scale-invariant accuracy: the multiplier scale tracks
        % ||grad||, which varies by orders of magnitude across problems and
        % shrinks as a PDE mesh refines.  (Ruiz equilibration above already made
        % the active block A(eqset,:) O(1)-scaled.)  NNLS is positively
        % homogeneous in the RHS, so q and the residual d rescale by s.
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
