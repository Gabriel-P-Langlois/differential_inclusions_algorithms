function [x,p,fval,flag] = di_rlp(A,b,c,t,x,tol,maxit,nnls_solver)
%   DI_RLP              Differential inclusions approach to solving
%                       L2-regularized linear programs.
%
%   This function solves the following feasible regularized linear program
%
%   (1) min_{x \in \Rn} (t/2)*\|x\|^2 + \langle c,x \rangle
%                       subject to A*x \leqslant b,
%
%   where t > 0 is a given regularization parameter.
%
%   The differential inclusions approach is a finite-time algorithm that
%   computes the continuous-time solution of the differential inclusions
%   associated to the problem. The solution consists of finitely many
%   piecewise continuous components calculated from a sequence of
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
%       t           -   Positive scalar regularization parameter
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
%                       Default is @apgd_lsqnonneg.
%
%   OUTPUTS
%       x           -   n-dimensional solution vector.
%                       If infeasible, returns empty field.
%       p           -   m-dimensional dual solution vector (KKT multipliers
%                       for the inequality constraints). If infeasible,
%                       returns empty field.
%       fval        -   Value of the objective function at the solution.
%       flag        -   Returns 0 if x is infeasible, returns -1 if the
%                       outer loop reached maxit iterations (tolerance issue
%                       or exponential behavior), and returns 1 if
%                       successful.
%
% -------------------------------------------------------------------------
%   QUICK EXAMPLE
      % % Generate the data
      % m = 200;
      % n = 400;
      % t = 1.0;
      % tol = 1e-8;
      % x0     = randn(n, 1);
      % A      = randn(m, n);
      % s      = abs(randn(m, 1)) + 1;
      % b      = A * x0 + s;
      % lambda = abs(randn(m, 1)) + 1;
      % c      = -A' * lambda;
      % 
      % opts = optimoptions('quadprog', 'Display', 'none');
      % [x_qp, fval_qp] = quadprog(t*eye(n), c, A, b, [], [], [], [], [], opts);
      % [x_di, p_di, fval_di, ~] = di_rlp(A, b, c, t, x0, tol, [], @apgd_lsqnonneg);
      % 
      % fprintf('primal gap:          %e\n', abs(fval_qp - fval_di))
      % fprintf('stationarity:        %e\n', norm(t*x_di + c + A'*p_di, inf))
      % fprintf('dual feasibility:    %e\n', max(0, -min(p_di)))
      % fprintf('complementary slack: %e\n', norm(p_di .* (b - A*x_di), inf))
%
% -------------------------------------------------------------------------


%% Preliminary checks
% Check for an acceptable number of input arguments
if nargin < 4
    error('di_rlp: Not enough input arguments.')
end
if ~iscolumn(b)
    error('di_rlp: The data vector b is not a column vector.');
end
if ~iscolumn(c)
    error('di_rlp: The data vector c is not a column vector.');
end


% Check matrix and vector inputs have appropriate sizes
[m,n] = size(A);
if size(b,1) ~= m
    error(['di_rlp: The number of rows of A does not match the ' ...
        'length of the column vector b.'])
end
if size(c,1) ~= n
    error(['di_rlp: The number of columns of A does not match the ' ...
        'length of the column vector c.'])
end


% Validate t
if ~isscalar(t) || t <= 0
    error('di_rlp: t must be a positive scalar.');
end


% Set defaults before the feasibility phase, which may need them.
if nargin < 6 || isempty(tol)
    tol = min(1e-08, 100*eps*max(norm(A,'fro'), 1)*max(norm(b), 1));
end
if nargin < 7 || isempty(maxit)
    maxit = max(1e6, 100*n);
end
if nargin < 8 || isempty(nnls_solver)
    nnls_solver = @apgd_lsqnonneg;
end

% Compute a feasible starting point if one was not supplied.
if nargin < 5 || isempty(x)
    [x, flag_feas] = di_phase1(A, b, tol, maxit, nnls_solver);
    if((flag_feas == 0) || (flag_feas == -1))
        fprintf('The linear program is infeasible within tolerance.\n')
        x = [];
        flag = 0;
        return;
    end
end


% Check if the supplied initial point is feasible.
fval = 0.5*t*(x'*x) + c'*x;
p = zeros(m,1);
flag = 1;
A_times_x = A*x;
residual = b - A_times_x;
if any(-residual > tol)
    fprintf('The initial point is infeasible within tolerance.\n')
    x = [];
    flag = 0;
    return;
end


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
            [q,d,veff] = nnls_solver(A(eqset,:).', -(t*x + c), tol, ...
                p(eqset), v(eqset));
            v(eqset)  = veff;
            v(~eqset) = 0;
        else
            [q,d] = nnls_solver(A(eqset,:).', -(t*x + c), tol, p(eqset));
        end
        p(eqset)  = q;
        p(~eqset) = 0;
        d = -d;
    else
        d = -(t*x + c);
    end

    % Compute the maximal descent time
    A_times_d = A*d;
    tmp = residual ./ A_times_d;
    tmp(eqset | (A_times_d <= 0)) = inf;
    timestep = min(tmp);

    % Check for convergence: unconstrained minimizer along d is at step 1/t.
    % If the first constraint is hit no sooner than 1/t, take the step to
    % the unconstrained minimizer and update the dual variable.
    if t * timestep >= 1
        x         = x + d/t;
        A_times_x = A_times_x + (1/t)*A_times_d;
        residual  = b - A_times_x;
        eqset     = abs(residual) < tol;
        neff      = sum(eqset);
        if neff > 0
            if use_v
                [q,~,veff] = nnls_solver(A(eqset,:).', -(t*x + c), tol, ...
                    p(eqset), v(eqset));
                v(eqset)  = veff;
                v(~eqset) = 0;
            else
                [q,~] = nnls_solver(A(eqset,:).', -(t*x + c), tol, p(eqset));
            end
            p(eqset)  = q;
            p(~eqset) = 0;
        else
            p = zeros(m,1);
        end
        fval = 0.5*t*(x'*x) + c'*x;
        return
    end

    % Update all quantities
    x         = x + timestep*d;
    A_times_x = A_times_x + timestep*A_times_d;
    residual  = b - A_times_x;
    eqset     = abs(residual) < tol;
    neff      = sum(eqset);
    fval      = 0.5*t*(x'*x) + c'*x;

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
