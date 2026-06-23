function [x,d] = pcg_lsqnonneg(A,b,tol,x)
%   PCG_LSQNONNEG   Projection Conjugate Gradient approach to
%                   solving the nonnegative least-squares (NNLS) problem.
%
%   Computes an optimal solution to
%
%       min_{x >= 0} (1/2)||A*x - b||_2^2
%
%   via the PCG algorithm of Moré and Toraldo (SIAM J. Optim., 1(1):93-113,
%   1991). Each outer iteration has two phases: gradient projection identifies
%   the active face, and conjugate gradient minimizes the reduced quadratic on
%   that face.
%
% -------------------------------------------------------------------------
%   INPUTS
%       A     - (mA x n)-dimensional matrix.
%       b     - mA-dimensional column data vector.
%       tol   - (Optional) Tolerance. Default: 1e-08.
%       x     - (Optional) Initial feasible point x >= 0.
%
%   OUTPUTS
%       x     - n-dimensional solution vector.
%       d     - mA-dimensional residual d = A*x - b.
%
% -------------------------------------------------------------------------
%   QUICK EXAMPLE
%       rng(1);
%       m = 200; n = 2000;
%       A = randn(m,n); b = randn(m,1);
%       [x_gpcg,d] = gpcg_lsqnonneg(A,b);
%
% -------------------------------------------------------------------------
%   NOTES
%       1) Each GP step requires two matrix-vector products with A and A'.
%       Each CG step requires two products with the reduced matrix A_F and
%       A_F'. The projected line search is O(n).
%
%       2) The algorithm does not implement the "continue CG" warm-restart
%       when B(x_{k+1}) = A(x_{k+1}): the outer loop handles this naturally
%       because the GP phase is cheap when the face is already identified.
%
%       3) The postprocessing step refines the solution on the identified
%       support via lsqr, identical to lbfgs_lsqnonneg.
%
% -------------------------------------------------------------------------


%% Preliminary checks
if nargin < 2
    error('gpcg_lsqnonneg: Not enough input arguments.')
end
if ~iscolumn(b)
    error('gpcg_lsqnonneg: b must be a column vector.');
end
[mA,n] = size(A);
if size(b,1) ~= mA
    error('gpcg_lsqnonneg: Row count of A does not match length of b.')
end
if nargin < 3 || isempty(tol)
    tol = min(1e-08, 100*eps*max(norm(A,'fro'),1)*max(norm(b),1));
end
if nargin < 4 || isempty(x)
    if issparse(A)
        x = sparse(n,1);
    else
        x = zeros(n,1);
    end
end
if min(x) < 0
    x = max(0,x);
end


%% Parameters (Moré-Toraldo 1991, §7)
eta2   = 0.25;          % GP sufficient-progress threshold  (eq. 3.8)
kmax   = 200000;
gp_max = min(n, 1000);
cg_max = min(n, 1000);


%% Algorithm
r = A*x - b;
g = A.'*r;

Free_prev = false(n,1);
if issparse(A)
    AF = sparse(mA,0);
else
    AF = zeros(mA,0);
end

for k = 1:kmax

    % Convergence check: ||P(x - g) - x||_inf < tol
    if norm(max(0, x - g) - x, 'inf') < tol
        break
    end

    % ------------------------------------------------------------------
    % Gradient projection phase: identify the active face.
    % ------------------------------------------------------------------
    x_prev = x;
    y = x;  ry = r;  gy = g;
    best_gp_dec = 0;
    act_prev    = (y == 0);

    for j = 1:gp_max

        % Projected gradient pg = nabla_Omega q(y): eq. (2.3)
        pg = gy;
        pg(y == 0) = min(gy(y == 0), 0);

        % Exact step size alpha_exact = ||pg||^2 / ||A*pg||^2: eq. (4.4)
        Apg     = A * pg;
        denom   = Apg.' * Apg;
        if denom < eps
            break
        end
        pgnorm2 = pg.'*pg;

        % First breakpoint: smallest alpha > 0 where y - alpha*g_y hits 0
        % (only at components with g_i > 0 and y_i > 0)
        pos_mask = (gy > 0) & (y > 0);
        if any(pos_mask)
            beta1_gp = min(y(pos_mask) ./ gy(pos_mask));
        else
            beta1_gp = inf;
        end

        alpha_gp = min(pgnorm2 / denom, beta1_gp);

        % Update: y_new = y - alpha*pg (exact for alpha <= beta1; eq. 4.2)
        y_new  = max(0, y  - alpha_gp * pg);
        ry_new =        ry - alpha_gp * Apg;
        gy_new =        gy - alpha_gp * (A.'*Apg);

        % dec_gp computed analytically: uses ry'*Apg = gy'*pg = ||pg||^2
        dec_gp = alpha_gp*pgnorm2 - 0.5*alpha_gp^2*denom;

        % Stopping conditions (3.7) and (3.8)
        act_cur = (y_new == 0);
        stop_37 = isequal(act_cur, act_prev);
        stop_38 = (j > 1) && (dec_gp <= eta2 * best_gp_dec);

        best_gp_dec = max(best_gp_dec, dec_gp);
        act_prev    = act_cur;
        y  = y_new;  ry = ry_new;  gy = gy_new;

        if stop_37 || stop_38
            break
        end
    end

    x = y;  r = ry;  g = gy;

    % Stagnation check after GP: if GP moved x by less than tol, the
    % iteration is stuck (degenerate active set near convergence).
    if norm(x - x_prev, 'inf') < tol
        break
    end

    % ------------------------------------------------------------------
    % CG phase: minimize the reduced quadratic on the free face.
    % ------------------------------------------------------------------
    Free = (x > 0);
    nf   = sum(Free);
    if nf == 0
        continue
    end
    if ~isequal(Free, Free_prev)
        AF        = A(:, Free);
        Free_prev = Free;
    end
    g_red = g(Free);        % reduced gradient = nabla q_k(0) in free coords

    % CG on  min_{w in R^nf} (1/2) w'(AF'AF)w + g_red'w,  w_0 = 0.
    % Aw = AF*w is maintained incrementally to avoid a matvec at line search.
    w   = zeros(nf, 1);
    Aw  = zeros(mA, 1);
    res = g_red;
    p   = -res;
    rr  = res.'*res;

    for j = 1:cg_max
        AFp = AF * p;
        pHp = AFp.'*AFp;    % p'*(AF'AF)*p = ||AF*p||^2
        if pHp < eps
            break
        end

        alpha_cg = rr / pHp;

        w      = w  + alpha_cg * p;
        Aw     = Aw + alpha_cg * AFp;     % AF*w, maintained at O(mA) per step
        res    = res + alpha_cg * (AF.'*AFp);
        rr_new = res.'*res;

        if sqrt(rr_new) < tol
            break
        end

        p  = -res + (rr_new / rr) * p;  % Fletcher-Reeves update
        rr = rr_new;
    end

    gd = g_red.'*w;  % directional derivative g'*dk = g(Free)'*w
    if gd >= 0
        continue
    end

    % Projected line search: exact minimizer on first quadratic piece.
    % Aw = AF*w is already available; no extra matvec needed.
    denom = Aw.'*Aw;
    if denom < eps
        continue
    end
    alpha_exact = -gd / denom;

    % First breakpoint in the CG direction (eq. 3.5)
    x_free = x(Free);
    neg_w  = (w < 0);
    if any(neg_w)
        beta1_cg = min(x_free(neg_w) ./ (-w(neg_w)));
    else
        beta1_cg = inf;
    end

    alpha_k = min(alpha_exact, beta1_cg);

    % Update (residual update is exact for alpha_k <= beta1_cg)
    x(Free)  = max(0, x_free + alpha_k * w);
    r        = r + alpha_k * Aw;
    g        = A.'*r;
end

[x,d] = postprocess_sol(A, b, x, x > tol);
end


%% Local functions

function [x,d] = postprocess_sol(A, b, x, eqcset)
    warning('off', 'MATLAB:lsqr:tooSmallTolerance')
    x(~eqcset) = 0;
    [x(eqcset),~] = lsqr(A(:,eqcset), b, eps, 1000, [], [], x(eqcset));
    d = A*x - b;
    warning('on', 'MATLAB:lsqr:tooSmallTolerance')
end
