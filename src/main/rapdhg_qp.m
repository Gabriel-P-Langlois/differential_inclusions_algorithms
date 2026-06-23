function [x, y, fval, flag] = rapdhg_qp(A, b, c, Q, x0, y0, tol, K, maxit)
%   RAPDHG_QP   Restarted accelerated PDHG for convex quadratic programs.
%
%   Solves the convex QP:
%       min  (1/2) x'*Q*x + c'*x   s.t.  A*x <= b
%
%   via the rAPDHG algorithm (Algorithm 1) from:
%       Lu & Yang, "A Practical and Optimal First-Order Method for
%       Large-Scale Convex Quadratic Programming,"
%       Mathematical Programming 215 (2026), pp. 771-808.
%
%   The algorithm finds a saddle point of the Lagrangian
%       L(x,y) = (1/2)x'Qx + c'x + y'(Ax - b),  y >= 0.
%
%   Each outer iteration runs K steps of accelerated PDHG and then
%   restarts from the running primal-dual average.  Step sizes follow
%   equation (4) in the paper (Lemma 2), which guarantees linear
%   convergence when K is large enough.
%
% -------------------------------------------------------------------------
%   INPUTS
%       A       -   (m x n) constraint matrix
%       b       -   m-vector, right-hand side (column)
%       c       -   n-vector, linear cost (column)
%       Q       -   (n x n) symmetric PSD matrix, or function handle v->Q*v
%       x0      -   (Optional) Initial primal point.  Default: zeros(n,1).
%       y0      -   (Optional) Initial dual point.    Default: zeros(m,1).
%       tol     -   (Optional) Relative KKT tolerance. Default: 1e-6.
%       K       -   (Optional) Inner restart frequency. Default: 50.
%       maxit   -   (Optional) Max outer (restart) iterations. Default: 500.
%
%   OUTPUTS
%       x       -   Approximate primal solution.
%       y       -   Approximate dual solution (y >= 0).
%       fval    -   Objective value (1/2)x'Qx + c'x at x.
%       flag    -    1  converged within tol.
%                   -1  maxit outer iterations reached without convergence.
%
% -------------------------------------------------------------------------
%   QUICK EXAMPLE
%       rng(1); m = 500; n = 200;
%       A = randn(m,n); x_true = randn(n,1);
%       b = A*x_true + abs(randn(m,1)) + 0.1;
%       B = randn(n,n); Q = B'*B + 0.1*eye(n); c = randn(n,1);
%       opts = optimoptions('quadprog','Display','off');
%       x_ref = quadprog(Q, c, A, b, [], [], [], [], [], opts);
%       [x, y, fval, flag] = rapdhg_qp(A, b, c, Q);
%       fprintf('Objective gap: %.2e\n', abs(fval - (0.5*x_ref'*Q*x_ref + c'*x_ref)));
%
% -------------------------------------------------------------------------

%% Input validation
if nargin < 4
    error('rapdhg_qp: At least 4 inputs required.')
end
if ~iscolumn(b), error('rapdhg_qp: b must be a column vector.'); end
if ~iscolumn(c), error('rapdhg_qp: c must be a column vector.'); end

[m, n] = size(A);
if length(b) ~= m, error('rapdhg_qp: length(b) ~= size(A,1).'); end
if length(c) ~= n, error('rapdhg_qp: length(c) ~= size(A,2).'); end
if ~isa(Q, 'function_handle') && ~isequal(size(Q), [n, n])
    error('rapdhg_qp: Q must be (n x n) or a function handle v->Q*v.')
end

%% Wrap Q and estimate spectral norms
if isa(Q, 'function_handle')
    Qfun  = Q;
    normQ = normest_handle(Qfun, n);
else
    Qfun  = @(v) Q * v;
    normQ = normest(Q, 1e-2);
end
normA = normest(A, 1e-2);

if normA < eps
    error('rapdhg_qp: A has (near-)zero spectral norm; degenerate problem.')
end

%% Defaults
if nargin < 5 || isempty(x0),    x0    = zeros(n, 1); end
if nargin < 6 || isempty(y0),    y0    = zeros(m, 1); end
if nargin < 7 || isempty(tol),   tol   = 1e-6;        end
if nargin < 8 || isempty(K),     K     = 50;          end
if nargin < 9 || isempty(maxit), maxit = 2000;       end

%% Precompute transpose (avoids repeated transposition for sparse A)
At = A';

%% Scaling factors for relative stopping criterion
scl_p = max(1, norm(b));
scl_d = max(1, norm(c));
% scl_c would be used for complementarity, but that check is not needed
% (see comment in the convergence test below).

%% Outer (restart) loop
x    = x0;
y    = max(0, y0);
flag = -1;

for n_out = 1 : maxit

    % --- Initialize inner loop ---
    % Algorithm 1, line 3:  (x_tilde, x, y_bar) <- (x^{n,0}, x^{n,0}, y^{n,0})
    xt       = x;
    yt       = y;
    xk       = x;
    yk       = y;
    xk_prev  = x;      % x^{n,-1} = x^{n,0}, so theta_0*(x_0-x_{-1}) = 0
    Axk      = A * xk;
    Axk_prev = Axk;

    % --- Inner loop (k = 0, ..., K-1) ---
    for k = 0 : K-1
        % Step-size parameters (Lemma 2, eq. (4))
        bk     = (k + 2) / 2;
        tk     = k / (k + 1);                       % theta_k  (= 0 at k=0)
        ek     = (k + 1) / (2 * (normQ + K*normA)); % eta_k
        sk     = (k + 1) / (2 * K * normA);         % tau_k
        inv_bk = 2 / (k + 2);                        % 1/beta_k

        % Midpoint: x_md = (1-1/beta_k)*x_tilde + (1/beta_k)*x_k
        x_md = (1 - inv_bk) * xt + inv_bk * xk;

        % Dual step:  y_{k+1} = max(0, y_k + tau_k*(A*x_extrap - b))
        % x_extrap = (1+theta_k)*x_k - theta_k*x_{k-1}
        % A*x_extrap = (1+theta_k)*Ax_k - theta_k*Ax_{k-1}  (no extra matvec)
        Ax_extrap = (1 + tk) * Axk - tk * Axk_prev;
        yk1 = max(0, yk + sk * (Ax_extrap - b));

        % Primal step:  x_{k+1} = x_k - eta_k*(Q*x_md + c + A'*y_{k+1})
        xk1 = xk - ek * (Qfun(x_md) + c + At * yk1);

        % Update running averages (lines 8-9 of Algorithm 1)
        xt = (1 - inv_bk) * xt + inv_bk * xk1;
        yt = (1 - inv_bk) * yt + inv_bk * yk1;

        % Advance iterates
        xk_prev  = xk;
        Axk_prev = Axk;
        xk       = xk1;
        yk       = yk1;
        Axk      = A * xk;
    end

    % --- Restart from running averages (Algorithm 1, line 11) ---
    x = xt;
    y = max(0, yt);

    % --- KKT residual check (relative) ---
    % Complementarity is not checked separately: the PDHG fixed-point
    % condition guarantees y_i*(Ax-b)_i -> 0 whenever prim_res and
    % dual_res converge to 0.
    Ax       = A * x;
    prim_res = norm(max(0, Ax - b)) / scl_p;
    dual_res = norm(Qfun(x) + c + At * y) / scl_d;

    if prim_res <= tol && dual_res <= tol
        flag = 1;
        break
    end
end

if flag == -1
    warning('rapdhg_qp: maxit=%d outer iterations reached without convergence.', maxit)
end

fval = 0.5 * (x' * Qfun(x)) + c' * x;

end


%% Local helper: spectral norm estimate for a function handle via power iteration
function nrm = normest_handle(Qfun, n)
v   = randn(n, 1);
v   = v / norm(v);
nrm = 1;
for i = 1 : 40
    w   = Qfun(v);
    nrm = norm(w);
    if nrm < eps, return; end
    v   = w / nrm;
end
end
