function [x, p, mu, fval, flag, fgap, iter] = di_qp_eq(A, b, c, Q, E, f, x, ...
    tol, maxit, PE_fun, EEt_solve_fun, nnls_solver, tol_nnls)
%   DI_QP_EQ    Differential inclusions approach to approximately solving
%               convex quadratic programs with equality and inequality
%               constraints.
%
%   This function computes an approximate optimal solution of
%
%   (1) min_{x in R^n}  (1/2)*x'*Q*x + c'*x
%                       subject to  A*x <= b,  E*x = f,
%
%   where Q is symmetric positive definite and E has full row rank.
%
%   The algorithm follows the differential inclusion associated to (1)
%   projected onto null(E) at each step. On each local interval the descent
%   direction is
%
%       d = -P_E(Q*x + c + A_eq'*p),
%
%   where P_E = I - E'*(E*E')^{-1}*E is the orthogonal projector onto
%   null(E), A_eq = A(eqset,:) is the active inequality block, and p >= 0
%   solves the projected NNLS subproblem
%
%       min_{p >= 0}  (1/2)*||P_E(A_eq'*p + Q*x + c)||^2.
%
%   Because d lies in null(E), the equality constraint E*x = f is preserved
%   along the trajectory. The step size is the minimum of the time to hit
%   the nearest inactive constraint and the exact quadratic line search time
%   ||d||^2 / (d'*Q*d).
%
%   The internal NNLS solver is a self-contained FISTA-based method (see
%   local function apgd_proj below), accessed entirely through two function
%   handles for the forward and adjoint operators. This allows PE_fun to be
%   supplied by the user when E has special structure (e.g., a PDE operator)
%   that makes the default Cholesky-based projector inefficient.
%
%   Unlike the LP and scalar-Q cases, finite termination is not guaranteed
%   in general; the algorithm converges as a descent method.
%
% -------------------------------------------------------------------------
%   INPUTS
%       A        -  (m x n) inequality constraint matrix
%       b        -  m-dimensional RHS column vector
%       c        -  n-dimensional cost column vector
%       Q        -  (n x n) symmetric positive definite matrix, or a
%                   function handle of the form v -> Q*v
%       E        -  (me x n) equality constraint matrix; must have full row
%                   rank (equivalently, E*E' must be positive definite)
%       f        -  me-dimensional RHS column vector
%       x        -  n-dimensional feasible starting point satisfying
%                   A*x <= b and E*x = f within tolerance; required (no
%                   Phase I is performed inside this function)
%       tol      -  (Optional) Tolerance. Default is
%                   min(1e-8, 100*eps*||A||_F * ||b||).
%       maxit    -  (Optional) Maximum number of outer iterations. Default
%                   is max(1e6, 100*n).
%       PE_fun       -  (Optional) Function handle v -> P_E*v, where
%                       P_E = I - E'*(E*E')^{-1}*E is the orthogonal
%                       projector onto null(E).  If absent, built
%                       internally via Cholesky of E*E'.  Supply this
%                       when E has special structure that makes the
%                       Cholesky impractical (e.g., a large PDE operator
%                       with a known factorization).
%       EEt_solve_fun - (Optional) Function handle r -> (E*E')^{-1}*r,
%                       used only to recover the equality dual mu at
%                       convergence.  Required when PE_fun is supplied and
%                       forming E*E' explicitly is impractical (e.g., a
%                       64^3 mesh in 3D, where N = 64^3 ~= 2.6e5 DOFs and
%                       E*E' = K^2 + M_c^2 has O(N^{5/3}) Cholesky fill).
%                       If absent and PE_fun is also absent, both are built
%                       from the same Cholesky of E*E'.
%       tol_nnls    -   (Optional) RELATIVE accuracy for the inner active-face
%                       NNLS subproblem, decoupled from the outer tolerance tol.
%                       The projected NNLS right-hand side P_E(-grad) is
%                       normalized to unit norm before each solve, so tol_nnls
%                       is a scale-invariant relative tolerance (see di_qp).  A
%                       loose NNLS returns a descent direction that violates the
%                       active constraints' optimality and the outer active-set
%                       path chatters.  Default is 1e-10.  Only the inner solve
%                       uses tol_nnls; membership and the ||d|| < tol stop keep
%                       tol.  NOTE: di_qp_eq does not equilibrate A, so the
%                       active block A_eq is left at its native scale; for box
%                       control bounds A_eq = +/- e_i is orthonormal, but a
%                       general badly-row-scaled inequality matrix would also
%                       need A_eq row-normalization (not done here).
%
%   OUTPUTS
%       x    -  n-dimensional approximate primal solution. Empty if
%               flag ~= 1.
%       p    -  m-dimensional inequality dual vector.
%       mu   -  me-dimensional equality dual vector.
%       fval -  Objective value (1/2)*x'*Q*x + c'*x at the solution.
%       flag -  Returns 1 if converged, -1 if the outer loop reached maxit
%               without converging, and 0 if the initial point is infeasible.
%       fgap -  Primal-dual gap p'*(b - A*x), the complementary slackness
%               residual.  Equals the objective gap (primal minus dual) when
%               stationarity holds at termination; zero at exact KKT.
%               Returns NaN if flag ~= 1.
%       iter -  (Optional) Number of outer iterations (active-set breakpoints)
%               performed.  Zero if the initial point is infeasible.
%
% -------------------------------------------------------------------------
%   QUICK EXAMPLE
      % % Generate data
      % rng(1);
      % m = 200; n = 100; me = 20;
      % x0 = randn(n, 1);
      % A  = randn(m, n);  b = A*x0 + abs(randn(m,1)) + 1;
      % E  = randn(me, n); f = E*x0;
      % B  = randn(n, n);  Q = B'*B + eye(n);  c = randn(n, 1);
      % tol  = 1e-8;
      % opts = optimoptions('quadprog', 'Display', 'off');
      % [~, fval_qp] = quadprog(Q, c, A, b, E, f, [], [], [], opts);
      % [~, ~, ~, fval_di, ~] = di_qp_eq(A, b, c, Q, E, f, x0, tol);
      % fprintf('fval gap: %e\n', abs(fval_qp - fval_di))
%
% -------------------------------------------------------------------------


%% Input validation

if nargin < 7
    error('di_qp_eq: At least 7 inputs required (A, b, c, Q, E, f, x).')
end
if ~iscolumn(b)
    error('di_qp_eq: b must be a column vector.')
end
if ~iscolumn(c)
    error('di_qp_eq: c must be a column vector.')
end
if ~iscolumn(f)
    error('di_qp_eq: f must be a column vector.')
end
if ~iscolumn(x)
    error('di_qp_eq: x must be a column vector.')
end

[m, n]   = size(A);
[me, nE] = size(E);

if length(b) ~= m
    error('di_qp_eq: Row count of A does not match length of b.')
end
if length(c) ~= n
    error('di_qp_eq: Column count of A does not match length of c.')
end
if nE ~= n
    error('di_qp_eq: Column count of E does not match column count of A.')
end
if length(f) ~= me
    error('di_qp_eq: Row count of E does not match length of f.')
end
if length(x) ~= n
    error('di_qp_eq: x must have length n.')
end
if ~isa(Q, 'function_handle') && ~isequal(size(Q), [n, n])
    error('di_qp_eq: Q must be an (n x n) matrix or a function handle v -> Q*v.')
end

% Wrap Q into a uniform function handle.
if isa(Q, 'function_handle')
    Qfun = Q;
else
    Qfun = @(v) Q * v;
end

% Set defaults.
if nargin < 8 || isempty(tol)
    tol = min(1e-8, 100*eps*max(norm(A, 'fro'), 1)*max(norm(b), 1));
end
if nargin < 9 || isempty(maxit)
    maxit = max(1e6, 100*n);
end
if nargin < 13 || isempty(tol_nnls)
    tol_nnls = 1e-09;     % relative NNLS accuracy (RHS normalized to unit norm)
end
if ~isnumeric(maxit)
    error('di_qp_eq: maxit must be a positive numeric scalar or [].')
end

% Build EEt_solve and PE_fun.
%
% When both PE_fun and EEt_solve_fun are supplied, skip forming E*E'
% entirely — this is critical for large 3D problems where the Cholesky
% of E*E' has O(N^{5/3}) fill and would not fit in RAM.
%
% When only PE_fun is supplied (but not EEt_solve_fun), we still need
% EEt_solve for mu recovery; warn the user and fall through to Cholesky.
%
% When neither is supplied, build both from a single Cholesky of E*E'.

have_pe          = nargin >= 10 && ~isempty(PE_fun);
have_eet         = nargin >= 11 && ~isempty(EEt_solve_fun);
use_direct_nnls  = nargin >= 12 && ~isempty(nnls_solver);

if have_pe && have_eet
    EEt_solve = EEt_solve_fun;
else
    if have_pe && ~have_eet
        warning(['di_qp_eq: PE_fun supplied but EEt_solve_fun is absent. ' ...
                 'Forming E*E'' and factoring via Cholesky for mu recovery. ' ...
                 'For large sparse E, supply EEt_solve_fun to avoid this.'])
    end
    EEt = E * E';
    if issparse(E)
        [L_chol, chol_status, perm_chol] = chol(EEt, 'lower', 'vector');
        if chol_status ~= 0
            error(['di_qp_eq: E*E'' is not positive definite. ' ...
                   'Check that E has full row rank.'])
        end
        EEt_solve = @(r) eet_solve_sparse(L_chol, perm_chol, me, r);
    else
        [R_chol, chol_status] = chol(EEt);
        if chol_status ~= 0
            error(['di_qp_eq: E*E'' is not positive definite. ' ...
                   'Check that E has full row rank.'])
        end
        EEt_solve = @(r) R_chol \ (R_chol' \ r);
    end
    if ~have_pe
        PE_fun = @(v) pe_apply(E, EEt_solve, v);
    end
end

% Check initial point feasibility.
A_times_x = A * x;
residual   = b - A_times_x;
if any(-residual > tol)
    fprintf('di_qp_eq: Initial point violates inequality constraints.\n')
    x = []; p = []; mu = []; fval = []; flag = 0; fgap = NaN; iter = 0;
    return
end
if norm(E*x - f, inf) > tol
    fprintf('di_qp_eq: Initial point violates equality constraints.\n')
    x = []; p = []; mu = []; fval = []; flag = 0; fgap = NaN; iter = 0;
    return
end


%% Algorithm
p      = zeros(m, 1);
v_nnls = zeros(m, 1);     % momentum warm-start vector for apgd_proj
flag   = 1;
eqset  = abs(residual) < tol;
neff   = sum(eqset);

% Gradient of the objective at the current iterate.
grad = Qfun(x) + c;

count = 0;
while true
    count = count + 1;

    % Solve the projected NNLS subproblem on the active inequality face.
    if neff > 0
        A_eq  = A(eqset, :);                             % neff x n
        Afun  = @(q) PE_fun(A_eq' * q);                 % R^neff -> R^n
        Atfun = @(r) A_eq * PE_fun(r);                  % R^n    -> R^neff
        b_nnls = PE_fun(-grad);                          % R^n

        % Normalize the projected RHS to unit norm so tol_nnls is a relative,
        % scale-invariant accuracy (see di_qp).  NNLS is positively homogeneous,
        % so q and d_res rescale by s.
        s      = max(norm(b_nnls), realmin);
        b_nnls = b_nnls / s;

        if use_direct_nnls
            % Form C column by column: C(:,j) = PE_fun(A_eq(j,:)').
            % Calling PE_fun on the full matrix A_eq' at once is incorrect
            % when EEt_solve uses the sparse path (eet_solve_sparse only
            % handles column vectors).
            C = zeros(n, neff);
            for jj = 1:neff
                C(:, jj) = PE_fun(A_eq(jj, :)');
            end
            [q, d_res]     = nnls_solver(C, b_nnls, tol_nnls, p(eqset)/s);
        else
            [q, d_res, v_out] = apgd_proj(Afun, Atfun, b_nnls, tol_nnls, neff, ...
                                           p(eqset)/s, v_nnls(eqset));
            v_nnls(eqset)  = v_out;
            v_nnls(~eqset) = 0;
        end
        q     = s * q;
        d_res = s * d_res;
        p(eqset)       = q;
        p(~eqset)      = 0;
        d = -d_res;    % d = -P_E(A_eq'*q + grad), lies in null(E)
    else
        d = -PE_fun(grad);
    end

    % Convergence: ||d|| < tol is the projected KKT stationarity condition.
    if norm(d) < tol
        fval = 0.5*(x' * Qfun(x)) + c'*x;

        % KKT gives mu = -(EE')^{-1} E*(grad + A_eq'*p).
        rhs_mu = E * (grad + A(eqset,:)' * p(eqset));
        mu     = -EEt_solve(rhs_mu);

        % Primal-dual gap p'*(b - A*x) = complementary slackness residual.
        % Equals primal minus dual objective when stationarity holds.
        fgap = p' * (b - A*x);
        iter = count;
        return
    end

    % Constraint-hitting time for inactive inequalities.
    A_times_d = A * d;
    
    % NNLS optimality requires A(eqset,:)*d <= 0; finite tolerance
    % can flip individual rows.  Project the repair back into null(E) so d
    % stays in the equality-feasible subspace.
    bad = eqset & (A_times_d > 0);
    if any(bad)
        A_bad = A(bad,:);
        d = d - PE_fun(A_bad' * (A_times_d(bad) ./ sum(A_bad.^2, 2)));
        A_times_d = A * d;
    end
    tmp = residual ./ A_times_d;
    tmp(eqset | (A_times_d <= 0)) = inf;
    t_constr = min(tmp);

    % Exact quadratic line search time along d.
    Qd     = Qfun(d);
    t_quad = (d' * d) / (d' * Qd);

    % Take the smaller of the two times.
    timestep = min(t_constr, t_quad);

    % Update.
    x         = x + timestep * d;
    grad      = grad + timestep * Qd;
    A_times_x = A_times_x + timestep * A_times_d;
    residual  = b - A_times_x;
    eqset     = abs(residual) < tol;
    neff      = sum(eqset);

    if count == maxit
        disp(['Warning: di_qp_eq outer loop did not converge. ' ...
              'Retry with a looser tolerance or a larger maxit.'])
        x = []; p = []; mu = []; fval = []; flag = -1; fgap = NaN; iter = count;
        return
    end
end


end

%% Local functions

% Apply P_E = I - E'*(E*E')^{-1}*E using the EEt_solve function handle.
function out = pe_apply(E, EEt_solve, v)
    out = v - E' * EEt_solve(E * v);
end


% Solve (E*E')*z = r using the sparse Cholesky factor (permuted lower form).
% Factorization convention: EEt(perm, perm) = L * L'.
function z = eet_solve_sparse(L, perm, me, r)
    z        = zeros(me, 1);
    z(perm)  = L' \ (L \ r(perm));
end


% NNLS solver: min_{q >= 0} (1/2)*||Afun(q) - b||^2
% via FISTA with gradient restart (operator-based version of apgd_lsqnonneg).
%
%   Afun  : R^nq -> R^nb   (forward operator)
%   Atfun : R^nb -> R^nq   (adjoint operator)
%   b     : nb-dimensional RHS
%   tol   : convergence tolerance
%   nq    : dimension of q (cannot be inferred from Afun)
%   q     : (Optional) warm-start for the primal variable
%   v     : (Optional) warm-start for the power-iteration vector
%
% Returns:
%   q     : nq-dimensional solution
%   d_res : nb-dimensional residual Afun(q) - b
%   v     : updated power-iteration vector (for reuse on the next call)
function [q, d_res, v] = apgd_proj(Afun, Atfun, b, tol, nq, q, v)
    if nargin < 6 || isempty(q)
        q = zeros(nq, 1);
    end
    if nargin < 7 || isempty(v)
        v = randn(nq, 1);
    end
    q = max(0, q);

    Atb  = -Atfun(b);
    kmax = 200000;

    [L_val, v] = warm_normest_fh(Afun, Atfun, v, nq);
    tau = 1 / L_val^2;

    qm = q;
    tk = 1.0;
    for k = 1:kmax                                          
        tkp   = 0.5*(1 + sqrt(1 + 4*tk^2));
        betak = (tk - 1) / tkp;
        yk    = q + betak*(q - qm);

        fwd = Afun(yk);
        Atd = Atfun(fwd);
        qkp = max(0, yk - tau*(Atd + Atb));

        % Gradient restart: undo momentum if it increased the objective.
        if (yk - qkp)' * (qkp - q) > 0
            yk   = q;
            fwd  = Afun(yk);
            Atd  = Atfun(fwd);
            qkp  = max(0, yk - tau*(Atd + Atb));
            tkp  = 1.0;
        end

        if norm(qkp - q) < tol*(1 + norm(qkp))
            q = qkp;
            break
        end

        qm = q;
        q  = qkp;
        tk = tkp;
    end
    [q, d_res] = postprocess_fh(Afun, Atfun, b, q, tol, nq);
end


% Estimate sigma_max(M_proj) via power iteration with function handles.
% Replaces warm_normest in apgd_lsqnonneg; only the two matvec lines differ.
function [L_val, v] = warm_normest_fh(Afun, Atfun, v, nq)
    e = norm(v);
    if e < 1e-6
        v = randn(nq, 1);
        e = norm(v);
    end
    cnt = 0;
    v   = v / e;
    e0  = 0;
    while abs(e - e0) > 1e-6 * e
        e0    = e;
        Av    = Afun(v);
        normAv = norm(Av);
        v     = Atfun(Av);
        normv = norm(v);
        e     = normv / normAv;
        v     = v / normv;
        cnt   = cnt + 1;
        if cnt > 50000
            disp('warm_normest_fh: Power iteration did not converge.')
            break
        end
    end
    L_val = e;
end


% Postprocess the NNLS solution on its free face via lsqr.
% Mirrors postprocess_sol in apgd_lsqnonneg but uses function handles.
function [q, d_res] = postprocess_fh(Afun, Atfun, b, q, tol_pp, nq)
    Free = q > tol_pp;
    q(~Free) = 0;
    if any(Free)
        afree = @(v, flag) afree_apply(Afun, Atfun, Free, nq, v, flag);
        warning('off', 'MATLAB:lsqr:tooSmallTolerance')
        [q_free, ~] = lsqr(afree, b, eps, 1000, [], [], q(Free));
        warning('on',  'MATLAB:lsqr:tooSmallTolerance')
        q(Free)  = q_free;
        q(~Free) = 0;
    end
    d_res = Afun(q) - b;
end


% Operator adapter for lsqr restricted to the free-face columns.
function out = afree_apply(Afun, Atfun, Free, nq, v, flag)
    if strcmp(flag, 'notransp')
        w       = zeros(nq, 1);
        w(Free) = v;
        out     = Afun(w);
    else
        out = Atfun(v);
        out = out(Free);
    end
end
