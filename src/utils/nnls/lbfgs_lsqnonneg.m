function [x,d] = lbfgs_lsqnonneg(A,b,tol,x)
%   LBFGS_LSQNONNEG  L-BFGS-B iterative approach to solving the
%                    nonnegative least-squares (NNLS) problem.
%
%   This function computes an optimal solution to the NNLS problem
%
%       min_{x \in \Rn} (1/2)\|A*x - b\|_2^2   subject to x \geqslant 0
%
%   via the L-BFGS-B algorithm of Byrd, Lu, Nocedal, and Zhu (SIAM J.
%   Sci. Comput., 16(5):1190-1208, 1995), using the generalized Cauchy
%   point (sec. 4) and the direct primal method for subspace minimization
%   (sec. 5.1). The Hessian of the objective is approximated by the
%   limited memory BFGS matrix B_k = theta*I - W_k*M_k*W_k'.
%
% -------------------------------------------------------------------------
%   INPUTS
%       A       -   (mA x n)-dimensional matrix.
%       b       -   mA-dimensional column data vector.
%       tol     -   (Optional) Small number specifying the tolerance
%                   (e.g., 1e-08). Default value is 1e-08.
%       x       -   (Optional) Initial feasible point x >= 0.
%
%   OUTPUTS
%       x       -   n-dimensional solution vector to the NNLS problem.
%       d       -   mA-dimensional residual vector d = A*x - b.
%
% -------------------------------------------------------------------------
%   QUICK EXAMPLE
%       rng(1);
%       tol = 1e-8;
%       m = 200; n = 2000;
%       A = randn(m,n); b = randn(m,1);
%       [x_lbfgs,d] = lbfgs_lsqnonneg(A,b);
%
% -------------------------------------------------------------------------
%   NOTES
%       1) This is an iterative algorithm, not an exact or active-set one.
%
%       2) Each outer iteration requires two matrix-vector products; this
%       is the dominant cost. Additional O(n*lmem) work arises from the
%       generalized Cauchy point and subspace minimization steps, where
%       lmem is the L-BFGS memory size.
%
%       3) The 2*lmem x 2*lmem linear system in the subspace step is a
%       negligible O(lmem^3) direct solve.
%
% -------------------------------------------------------------------------


%% Preliminary checks
if nargin < 2
    error('lbfgs_lsqnonneg: Not enough input arguments.')
end
if ~iscolumn(b)
    error('lbfgs_lsqnonneg: The data vector b is not a column vector.');
end

[mA,n] = size(A);
if size(b,1) ~= mA
    error(['lbfgs_lsqnonneg: The number of rows of A does not match ' ...
        'the length of the column vector b.'])
end

if nargin < 3 || isempty(tol)
    tol = min(1e-08,100*eps*max(norm(A, 'fro'), 1)*max(norm(b), 1));
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


%% Algorithm
lmem  = 10;
kmax  = 200000;

% L-BFGS correction pairs and scaling
S     = zeros(n,lmem);
Y     = zeros(n,lmem);
lcur  = 0;
theta = 1.0;

r = A*x - b;
g = A.'*r;

for k = 1:kmax

    % Convergence check: ||P(x - g, 0, inf) - x||_inf < tol
    if norm(max(0, x - g) - x, 'inf') < tol
        break;
    end

    % Build compact L-BFGS representation W, M
    [W,M] = build_lbfgs_WM(S(:,1:lcur), Y(:,1:lcur), theta);

    % Generalized Cauchy point (Byrd et al., sec. 4)
    [xc,c_vec,free] = lbfgs_cauchy(x, g, W, M, theta);

    % Subspace minimization, direct primal method (Byrd et al., sec. 5.1)
    x_tilde = lbfgs_subspace(x, g, xc, c_vec, W, M, theta, free);

    % Exact quadratic line search along dk = x_tilde - x
    dk    = x_tilde - x;
    Adk   = A*dk;
    gdk   = g.'*dk;
    denom = Adk.'*Adk;

    if denom < eps * (dk'*dk) || gdk >= 0
        break;
    end

    lambda = min(-gdk/denom, max_feasible_step(x, dk));
    if lambda <= 0
        break;
    end

    x_new = x + lambda*dk;
    r_new = r + lambda*Adk;
    g_new = A.'*r_new;

    % L-BFGS curvature update
    sk = x_new - x;
    yk = g_new - g;
    sy = sk.'*yk;
    if sy > eps*(yk.'*yk)
        if lcur < lmem
            lcur = lcur + 1;
        else
            S(:,1:end-1) = S(:,2:end);
            Y(:,1:end-1) = Y(:,2:end);
        end
        S(:,lcur) = sk;
        Y(:,lcur) = yk;
        theta = (yk.'*yk)/sy;
    end

    x = x_new;
    r = r_new;
    g = g_new;
end

[x,d] = postprocess_sol(A, b, x, x > tol, tol);
end


%% Local functions

function [W,M] = build_lbfgs_WM(S, Y, theta)
    lcur = size(S,2);
    if lcur == 0
        W = zeros(size(S,1), 0);
        M = zeros(0,0);
        return;
    end
    SY    = S.'*Y;
    d     = diag(SY);               % s_i'*y_i > 0 by curvature condition
    L     = tril(SY,-1);
    SS    = S.'*S;

    % Schur complement of -D: Sc = theta*SS + L*D^{-1}*L', which is SPD.
    Dinv  = 1 ./ d;                 % O(lcur), diagonal inverse
    LDinv = L .* Dinv.';            % L * D^{-1}, column scaling
    Sc    = theta*SS + LDinv*L.';

    % Invert Sc via Cholesky (SPD), then assemble M = B^{-1} by blocks.
    R     = chol(Sc);               % Sc = R'*R, O(lcur^3)
    ScInv = R \ (R' \ eye(lcur));
    M21   = ScInv * LDinv;          % Sc^{-1} * L * D^{-1}
    M11   = diag(-Dinv) + LDinv.' * M21;
    M     = [M11, M21.'; M21, ScInv];
    W     = [Y, theta*S];           % n x 2*lcur
end


function [xc,c_vec,free] = lbfgs_cauchy(x, g, W, M, theta)
% Generalized Cauchy point via Algorithm CP of Byrd et al. (1995),
% simplified for NNLS bounds l = 0, u = inf.

    l2 = size(W,2);

    % Breakpoints: t_i = x_i/g_i for g_i > 0, x_i > 0.
    % Search direction: d_i = -g_i if g_i > 0 and x_i > 0, or g_i < 0; else 0.
    has_bp = (g > 0) & (x > 0);
    d_vec  = -g .* (has_bp | (g < 0));

    c_vec = zeros(l2,1);

    if l2 > 0
        p  = W.'*d_vec;
        Mp = M*p;
        Mc = zeros(l2,1);
        fpp = theta*(d_vec.'*d_vec) - p.'*Mp;
    else
        p   = zeros(0,1);
        Mp  = zeros(0,1);
        Mc  = zeros(0,1);
        fpp = theta*(d_vec.'*d_vec);
    end

    fp = g.'*d_vec;

    if fpp <= 0
        xc   = x;
        free = x > 0;
        return;
    end
    dt_min = -fp/fpp;

    % Sort finite breakpoints in ascending order
    bp_idx              = find(has_bp);
    t_bp                = x(bp_idx) ./ g(bp_idx);
    [bp_times, si]      = sort(t_bp);
    bp_sorted_idx       = bp_idx(si);
    nb                  = numel(bp_idx);

    t_old = 0;
    j     = 1;
    if l2 == 0
        while j <= nb
            dt = bp_times(j) - t_old;
            if dt_min < dt
                break;
            end
            b   = bp_sorted_idx(j);
            gb  = g(b);
            fp  = fp + dt*fpp + gb^2 - theta*gb*x(b);
            fpp = fpp - 2*theta*gb^2;
            if fpp <= 0
                dt_min = 0;
                t_old  = bp_times(j);
                break;
            end
            dt_min = -fp/fpp;
            t_old  = bp_times(j);
            j      = j + 1;
        end
    else
        while j <= nb
            dt = bp_times(j) - t_old;
            if dt_min < dt
                break;
            end
            b    = bp_sorted_idx(j);
            z_b  = -x(b);
            gb   = g(b);
            Mc    = Mc + dt*Mp;
            c_vec = c_vec + dt*p;
            wb  = W(b,:).';
            Mwb = M*wb;
            fp  = fp + dt*fpp + gb^2 + theta*gb*z_b - gb*(wb.'*Mc);
            fpp = fpp - 2*theta*gb^2 - 2*gb*(wb.'*Mp) - gb^2*(wb.'*Mwb);
            Mp  = Mp + gb*Mwb;
            p   = p  + gb*wb;
            if fpp <= 0
                dt_min = 0;
                t_old  = bp_times(j);
                break;
            end
            dt_min = -fp/fpp;
            t_old  = bp_times(j);
            j      = j + 1;
        end
    end

    % Finalize GCP position
    dt_min  = max(dt_min, 0);
    c_vec   = c_vec + dt_min*p;
    t_final = t_old + dt_min;
    xc      = max(0, x - t_final*g);

    free = xc > 0;
end


function x_tilde = lbfgs_subspace(x, g, xc, c_vec, W, M, theta, free)
% Direct primal subspace minimization (Byrd et al., sec. 5.1).

    x_tilde = xc;
    t_free  = sum(free);
    if t_free == 0
        return;
    end

    l2 = size(W,2);

    % Reduced gradient at xc (eq. 5.4)
    if l2 > 0
        rhat_full = g + theta*(xc - x) - W*(M*c_vec);
    else
        rhat_full = g + theta*(xc - x);
    end
    rhat_free = rhat_full(free);

    if l2 == 0
        dhat_u = -(1/theta)*rhat_free;
    else
        % Sherman-Morrison-Woodbury solve (eqs. 5.10-5.11):
        % d^u = -B^{-1} r^c = -(1/theta)*r^c - (1/theta^2)*WF*N^{-1}*M*WF'*r^c
        WF = W(free,:);                              % t_free x l2
        v  = WF.'*rhat_free;                         % l2 x 1
        v  = M*v;
        N  = eye(l2) - (1/theta)*M*(WF.'*WF);       % l2 x l2; indefinite by design
        warning('off', 'MATLAB:nearlySingularMatrix');
        warning('off', 'MATLAB:singularMatrix');
        v  = N\v;
        warning('on',  'MATLAB:nearlySingularMatrix');
        warning('on',  'MATLAB:singularMatrix');
        dhat_u = -(1/theta)*rhat_free - (1/theta^2)*(WF*v);
    end

    % Feasibility clip: alpha* = max{alpha <= 1 : xc_free + alpha*dhat_u >= 0}
    xc_free = xc(free);
    neg     = dhat_u < 0;
    if any(neg)
        alpha = min(1, min(xc_free(neg) ./ (-dhat_u(neg))));
        alpha = max(alpha, 0);
    else
        alpha = 1;
    end

    x_tilde(free) = xc_free + alpha*dhat_u;
    x_tilde       = max(0, x_tilde);
end


function delta = max_feasible_step(x, d)
    neg = d < 0;
    if any(neg)
        delta = min(x(neg) ./ (-d(neg)));
    else
        delta = inf;
    end
end


function [x,d] = postprocess_sol(A, b, x, eqcset, ~)
    warning('off', 'MATLAB:lsqr:tooSmallTolerance')
    x(~eqcset) = 0;
    [x(eqcset),~] = lsqr(A(:,eqcset), b, eps, 1000, [], [], x(eqcset));
    x(eqcset) = max(0, x(eqcset));   % lsqr is unconstrained; enforce x >= 0
    d = A*x - b;
    warning('on', 'MATLAB:lsqr:tooSmallTolerance')
end
