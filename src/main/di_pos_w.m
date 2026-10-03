function [x,flag,nbp,used_fast,mu,p,fval] = di_pos_w(ubar,m,M,w,x,tol,maxit,fast)
%   DI_POS_W        Gradient inclusion solver for the weighted
%                   bound-preserving limiter.
%
%   Solves exactly, in finitely many breakpoints,
%
%   (1) min_{x in R^N}  (1/2)*||x - ubar||_W^2
%                       subject to  w'*x = w'*ubar,  m <= x <= M,
%
%   with W = diag(w) and w > 0.  With w = ones and M = +Inf this is the
%   cell-average limiter (24) of Liu-Hu-Taitano-Zhang (CAMWA 192, 2025).
%   The solver needs no parameter.
%
%   METHOD.  One diagonal whitening reduces (1) to the unweighted case: with
%   wt = sqrt(w), y = wt.*x and yb = wt.*ubar, the objective becomes
%   (1/2)*||y - yb||^2 under wt'*y = b and wt.*m <= y <= wt.*M, so Q = I and
%   the theory for Q = t*I applies.  The projector is
%   P*v = v - wt*(wt'*v)/||wt||^2, and the NNLS over the active bound rows
%   collapses to one scalar nu, computed exactly by the local function
%   nnls_threshold_w.  The direction is closed form, d = r + nu*wt off the
%   support and 0 on it, the exact line search lands at step 1, and
%   p = wt.*p_whitened maps the multipliers back.
%
%   FAST PATH.  When M = +Inf and the start is affine in ubar on its free
%   set (the clip-and-scale start and the default start both are), the
%   path of Algorithm 1 has closed form: held rows never leave, the free
%   points stay affine in ubar, and they reach the bound in increasing
%   order of ubar.  The local function affine_path_w follows the same
%   breakpoints with O(1) work per breakpoint, after a sort of the free
%   points that can reach the bound; see its comments.  Otherwise, or with
%   fast = false, the general loop runs.
%
% -------------------------------------------------------------------------
%   INPUTS
%       ubar          -   N-dimensional column vector of data.
%       m, M          -   Scalar bounds, m < M.  M may be +Inf; m must be
%                         finite.
%       w             -   N-dimensional column vector of positive weights.
%       x             -   (Optional) Feasible start: w'*x = w'*ubar and
%                         m <= x <= M.  Default is the constant field
%                         (w'*ubar/sum(w))*ones(N,1), feasible iff (1) is.
%       tol           -   (Optional) Feasibility tolerance of the start and
%                         of the returned point.
%       maxit         -   (Optional) Breakpoint cap.  Default is
%                         max(1e6, 100*N).
%       fast          -   (Optional) Use the fast path when it applies.
%                         Default is true; false forces the general loop.
%
%   OUTPUTS
%       x             -   Exact minimizer up to round-off; empty if the
%                         start is infeasible or maxit is reached.
%       flag          -   1 converged, 0 infeasible start, -1 reached maxit.
%       nbp           -   Breakpoints, the landing pass included.
%       used_fast     -   true if the fast path ran.
%       mu            -   Scalar equality dual.
%       p             -   2N-dimensional bound dual, rows [upper; lower]:
%                         W*(x - ubar) + mu*w + p(1:N) - p(N+1:2N) = 0.
%       fval          -   (1/2)*||x - ubar||_W^2.
%   The fast path forms p and fval only when they are requested.
%
% -------------------------------------------------------------------------


%% Preliminary checks
if nargin < 4
    error('di_pos_w: Not enough input arguments.')
end
if ~iscolumn(ubar) || ~iscolumn(w)
    error('di_pos_w: ubar and w must be column vectors.');
end
if ~isscalar(m) || ~isscalar(M) || m >= M
    error('di_pos_w: The bounds must be scalars with m < M.');
end
if any(w <= 0)
    error('di_pos_w: The weights must be positive.');
end

N = size(ubar,1);
b = w' * ubar;

% Range of the box.  With M = +Inf the minimizer lies in [m, max(ubar)].
if isinf(M)
    bnd_rng = max(ubar) - m;
else
    bnd_rng = M - m;
end

if nargin < 6 || isempty(tol)
    tol = min(1e-08, 100*eps*max(bnd_rng, 1)*max(norm(ubar), 1));
end
if nargin < 7 || isempty(maxit)
    maxit = max(1e6, 100*N);
end
if nargin < 8 || isempty(fast)
    fast = true;
end

% Active-set membership window, original units.  A breakpoint puts a
% coordinate exactly on its bound, so the window only absorbs round-off.
memb = 10*eps*max(bnd_rng, 1);

flag = 1;  nbp = 0;  used_fast = false;  mu = [];  p = [];  fval = [];

if nargin < 5 || isempty(x)
    x = (b / sum(w)) * ones(N,1);
end
if (~isinf(M) && any(x > M + tol)) || any(x < m - tol) || abs(w'*x - b) > tol*max(1, abs(b))
    warning('di_pos_w:infeasibleStart', ...
        'di_pos_w: the initial point is infeasible within tolerance.');
    x = [];  flag = 0;
    return
end


%% Fast path: M = +Inf, start affine in ubar on its free set
if fast && isinf(M)
    [ok, x_f, p_f, mu_f, nbp_f] = affine_path_w(ubar, m, w, x, memb, nargout >= 6);
    if ok
        x = x_f;  p = p_f;  mu = mu_f;  nbp = nbp_f;  used_fast = true;
        if nargout >= 7
            fval = 0.5 * sum(w .* (x - ubar).^2);
        end
        check_return(x, m, M, w, b, tol, memb);
        return
    end
end


%% Whitening
wt  = sqrt(w);
nw2 = sum(w);            % = ||wt||^2
yb  = wt .* ubar;
y   = wt .* x;
lo  = m * wt;
up  = M * wt;
mwin = memb * wt;        % y - lo < mwin  <=>  x - m < memb

eqU = y > up - mwin;
eqL = y < lo + mwin;

count = 0;
while true
    count = count + 1;

    % Projected negative gradient.
    rhs = yb - y;
    rhs = rhs - wt * ((wt' * rhs) / nw2);

    idx = find(eqU | eqL);
    if ~isempty(idx)
        sgn = ones(numel(idx),1);
        sgn(eqL(idx)) = -1;
        [nu, F] = nnls_threshold_w(rhs(idx), wt(idx), sgn, nw2);
        d = rhs + nu * wt;
        d(idx(F)) = 0;
    else
        d = rhs;
    end

    % Land directly when d lies inside the round-off window: the ratio test
    % is meaningless there and could cross the bound.
    if max(abs(d) ./ wt) <= memb
        timestep = Inf;
    else
        % Ratio test; the sign of d_i selects the reachable bound.  Ratios
        % toward a coordinate's own active bound are masked, since at
        % denormal scale d can land one ulp on the wrong side of zero.
        if isinf(M)
            tgt = lo;  tgt(d > 0) = up(d > 0);        % avoids Inf*0
        else
            tgt = lo + (up - lo) .* (d > 0);
        end
        ratio = (tgt - y) ./ d;
        ratio(d == 0) = inf;
        ratio((d > 0 & eqU) | (d < 0 & eqL)) = inf;
        timestep = min(ratio);
    end

    if timestep < 0
        warning('di_pos_w:negativeStep', ...
            'di_pos_w: negative step %.2e at iteration %d; clamped to zero.', ...
            timestep, count);
        timestep = 0;
    end

    % The unconstrained minimizer along d sits at step 1.
    if timestep >= 1
        y   = y + d;
        eqU = y > up - mwin;
        eqL = y < lo + mwin;

        % Multipliers from the final face.
        rhs = yb - y;
        gm  = (wt' * (y - yb)) / nw2;
        rhs = rhs + gm * wt;
        idx = find(eqU | eqL);
        pU = zeros(N,1);
        pL = zeros(N,1);
        if ~isempty(idx)
            sgn = ones(numel(idx),1);
            sgn(eqL(idx)) = -1;
            [nu, F] = nnls_threshold_w(rhs(idx), wt(idx), sgn, nw2);
            rowsF = idx(F);
            sgnF  = sgn(F);
            qF    = sgnF .* (rhs(rowsF) + wt(rowsF)*nu);   % whitened q >= 0
            pU(rowsF(sgnF > 0)) = wt(rowsF(sgnF > 0)) .* qF(sgnF > 0);
            pL(rowsF(sgnF < 0)) = wt(rowsF(sgnF < 0)) .* qF(sgnF < 0);
        else
            nu = 0;
        end
        mu  = -(gm + nu);
        p   = [pU; pL];
        x   = y ./ wt;
        nbp = count;
        if nargout >= 7
            fval = 0.5 * sum(w .* (x - ubar).^2);
        end
        check_return(x, m, M, w, b, tol, memb);
        return
    end

    y   = y + timestep*d;
    eqU = y > up - mwin;
    eqL = y < lo + mwin;

    if count == maxit
        warning('di_pos_w:maxit', ...
            'di_pos_w: no convergence within maxit = %d breakpoints.', maxit);
        x = [];  flag = -1;  nbp = count;
        return
    end
end

end


function check_return(x, m, M, w, b, tol, memb)
%   CHECK_RETURN  Warn if the returned point violates feasibility.
    xmin = min(x);
    viol = max(m - xmin, abs(w'*x - b)/max(1, abs(b)));   % conservation relative to |b|
    if ~isinf(M)
        viol = max(viol, max(x) - M);
    end
    if viol > 10*tol
        warning('di_pos_w:infeasibleReturn', ...
            'di_pos_w: returned point violates feasibility by %.2e.', viol);
    end
    if xmin < m - memb
        warning('di_pos_w:belowBound', ...
            'di_pos_w: min(x) is %.2e below the lower bound.', m - xmin);
    end
end


function [ok, x, p, mu, nbp] = affine_path_w(u, m, w, x0, memb, want_p)
%   AFFINE_PATH_W  Algorithm 1 of the paper for the positivity
%   limiter, M = +Inf, from a start x0 that is affine in u on its free set.
%
%   Write a = u - m, F the free set, A its complement (rows on the bound),
%   W_F = sum_F w, S_F = sum_F w.*u and ubF = S_F/W_F.  Then:
%     (1) Feasibility gives w'*(u - x) = 0, so the projected residual is
%         u - x, and the direction is d = (u - x) + nu on F, 0 on held rows.
%     (2) Held rows never leave.  If every row of A is held, then
%         nu = sum_A w.*a / W_F.  A point joins A only if a + nu < 0, and
%         then a + nu' = W_F*(a + nu)/(W_F - w_i) < 0; rows that joined
%         earlier have smaller a.  So the NNLS support is all of A, and no
%         NNLS is solved.
%     (3) If x = alpha + beta*u on F, then d = (1 - beta)*(u - ubF) on F.
%         After a step t, x = alpha' + beta'*u on F with
%         alpha' = alpha - t*(1 - beta)*ubF and beta' = beta + t*(1 - beta).
%     (4) The hitting time t_i = (alpha + beta*u_i - m)/((1-beta)*(ubF - u_i))
%         increases with u_i for 0 <= beta < 1, so the next point to hit
%         the bound is the smallest free u.  A step t >= 1 lands on the
%         minimizer of the current face.
%   The breakpoints and the minimizer are those of the general loop in
%   exact arithmetic.  Ties in u (within memb) move together.  x is formed
%   once, at the end, from the final face: x = u + nu on F, x = m on A.
%
%   CANDIDATES.  The loop reads the sorted free u in increasing order and
%   stops at the first value it does not move, so only a prefix is sorted.
%   Let D = sum(w.*max(m - u, 0)) be the deficit.  The minimizer is
%   max(u + nu*, m), with nu* the root of the nondecreasing function
%   g(nu) = sum(w.*max(u + nu, m)) - w'*u, and g(0) = D.  If tau > 0 and
%   the free points with u - m >= tau have total weight W, then
%   g(-tau) <= D - tau*W, so tau*W >= D gives nu* >= -tau: every point
%   on the bound at the minimizer has u - m <= tau.  One pass bins the
%   free u - m on the grid tau_k = 2^k*D/W_F and takes the smallest
%   certified tau_k; the free points below 2*tau_k are sorted, and the
%   rest are sorted only if the loop reads past them (round-off, or no
%   certified tau_k).  The loop then reads the same values as with a full
%   sort, so the breakpoints are unchanged.
%
%   ok = false, with the other outputs empty, if the start is not affine on
%   its free set, beta lies outside [0, 1], or a row of A is not held at
%   the start; the caller then runs the general loop.  p is formed only if
%   want_p is true.

    x = [];  p = [];  mu = [];  nbp = 0;
    % Rows on the bound at the start: x0 = m exactly, as the clip-and-scale
    % start leaves every clipped entry.  A point just above m is free.
    Fr = x0 > m;
    A  = ~Fr;
    if ~any(Fr)
        ok = false;
        return
    end

    % Affine start on F: fit alpha and beta at the extreme u, then check.
    uF = u(Fr);  xF = x0(Fr);  wF = w(Fr);
    [umin, imin] = min(uF);
    [umax, imax] = max(uF);
    if umax > umin
        beta = (xF(imax) - xF(imin)) / (umax - umin);
    else
        beta = 0;
    end
    alpha = xF(imin) - beta*umin;
    ok = beta >= 0 && beta <= 1 && max(abs(xF - (alpha + beta*uF))) <= 10*memb;

    % Every row of A held at the start: a_j + nu < 0.
    WF = sum(wF);
    SA = 0;                          % sum_A w.*a
    if ok && any(A)
        uA = u(A);
        SA  = sum(w(A) .* (uA - m));
        nu0 = SA / WF;
        ok  = all(uA - m + nu0 < 0);
    end
    if ~ok
        return
    end

    % Candidates (see CANDIDATES): free points with u - m < 2*tau_k.
    idxF = find(Fr);
    nF   = numel(idxF);
    eF   = uF - m;
    D    = -SA - sum(wF .* min(eF, 0));
    inC  = true(nF, 1);
    if D > 0
        tau0 = D / WF;
        [~, E] = log2(eF / tau0);            % eF/tau0 in [2^(E-1), 2^E)
        E(eF <= 0) = 0;
        Emax = max(E);
        if Emax >= 1
            Wbin = accumarray(max(E, 0) + 1, wF, [Emax+1 1]);   % bin 1: E <= 0
            Wge  = flipud(cumsum(flipud(Wbin(2:end))));     % Wge(j) = W(E >= j)
            kk   = find(tau0 * 2.^(0:Emax-1)' .* Wge >= D, 1);
            if ~isempty(kk)                  % tau_k = tau0*2^(kk-1)
                inC = E <= kk;               % eF < 2*tau_k
            end
        end
    end
    [us, o] = sort(uF(inC));
    iS = idxF(inC);   iS = iS(o);            % sorted global indices
    ws = wF(inC);     ws = ws(o);
    notC = ~inC;                             % the rest, sorted by extend()
    WR = sum(wF(notC));
    nC = numel(us);
    cw  = cumsum(ws);
    cwu = cumsum(ws .* us);
    SF  = sum(wF .* uF);

    % Breakpoints: scalar work only.  The next breakpoint belongs to the
    % smallest free u.  Every group whose u lies within memb of that value
    % moves with it, except the group of the largest free u.  The window is
    % anchored at the smallest value, as in the general loop.
    k = 0;                         % sorted points moved to the bound
    while beta < 1
        if k == nC, extend(); end
        u1 = us(k+1);
        if u1 >= umax, break; end  % only the largest group is left
        if k > 0
            WFg = WF - cw(k);  SFg = SF - cwu(k);
        else
            WFg = WF;  SFg = SF;
        end
        ubF = SFg / WFg;
        if u1 >= ubF, break; end   % no free point moves toward the bound
        t = (alpha + beta*u1 - m) / ((1 - beta)*(ubF - u1));
        if t >= 1, break; end      % the step lands on the face minimizer
        alpha = alpha - t*(1 - beta)*ubF;
        beta  = beta + t*(1 - beta);
        v = u1 + memb;
        if nC < nF && us(nC) <= v && us(nC) < umax, extend(); end
        k = last_in_window(us, k+1, nC, v, umax);
        nbp = nbp + 1;
    end

    % Final face, from fresh sums: x = u + nu on F, x = m on A.  If
    % round-off left a free point with u + nu < m, that point is the path's
    % next breakpoint: move its group to the bound and recompute.  The test
    % uses the same expression u + nu as x, so min(x) >= m on exit.
    while true
        nu = (SA + sum(ws(1:k) .* (us(1:k) - m))) / (sum(ws(k+1:nC)) + WR);
        if k == nC, extend(); end
        if us(k+1) < umax && us(k+1) + nu < m
            k = last_in_window(us, k+1, nC, us(k+1), umax);
            nbp = nbp + 1;
        else
            break
        end
    end
    nbp = nbp + 1;                 % the landing pass
    x  = u + nu;
    x(A) = m;
    x(iS(1:k)) = m;
    if want_p
        onB = A;  onB(iS(1:k)) = true;
        pL = zeros(numel(u),1);
        pL(onB) = w(onB) .* (-(u(onB) - m) - nu);   % = w.*(m - u - nu) >= 0
        p  = [zeros(numel(u),1); pL];
    end
    mu = -nu;

    function extend()
    %   Append the sorted remaining free points; they all lie above us.
        iR = idxF(notC);
        [uRs, o2] = sort(uF(notC));
        wRs = wF(notC);  wRs = wRs(o2);
        cw0 = 0;  cwu0 = 0;
        if nC > 0
            cw0 = cw(end);  cwu0 = cwu(end);
        end
        us  = [us; uRs];
        ws  = [ws; wRs];
        iS  = [iS; iR(o2)];
        cw  = [cw;  cw0  + cumsum(wRs)];
        cwu = [cwu; cwu0 + cumsum(wRs .* uRs)];
        notC(:) = false;  WR = 0;
        nC = numel(us);
    end
end


function j = last_in_window(us, lo, hi, v, umax)
%   LAST_IN_WINDOW  Largest j in [lo, hi] with us(j) <= v and us(j) < umax,
%   by bisection on the sorted us; us(lo) satisfies both.
    while lo < hi
        mid = floor((lo + hi + 1)/2);
        if us(mid) <= v && us(mid) < umax
            lo = mid;
        else
            hi = mid - 1;
        end
    end
    j = lo;
end


function [nu, F] = nnls_threshold_w(rE, wE, sgn, nw2)
%   NNLS_THRESHOLD_W  Exact weighted scalar-threshold NNLS on the active
%   rows (at least one).
%
%   Returns the root nu of
%       h(nu) = sum_{F(nu)} wE_i*rE_i - nu*(nw2 - sum_{F(nu)} wE_i^2)
%   and the support mask F (upper rows: rE_i + wE_i*nu > 0; lower rows:
%   rE_i + wE_i*nu < 0 -- one threshold on the ratios rE_i/wE_i since
%   wE_i > 0).  h is continuous and strictly decreasing as long as the
%   active weights satisfy sum_F wE_i^2 < nw2; walk the sorted crossings
%   -rE_i/wE_i, evaluating h exactly at each; at the first h <= 0 the root
%   lies in the current piece, nu = S/(nw2 - K); if h stays positive the
%   root lies in the last, unbounded piece.  At each crossing an upper row
%   enters the support or a lower row leaves it, updating the weighted
%   running sums S, K, and the mask.

    F = (sgn < 0);
    S = sum(wE(F) .* rE(F));
    K = sum(wE(F).^2);

    [cr, ord] = sort(-rE ./ wE);
    nu = [];
    for j = 1:numel(rE)
        if S - cr(j)*(nw2 - K) <= 0
            nu = S / (nw2 - K);
            break
        end
        i = ord(j);
        if sgn(i) > 0
            F(i) = true;   S = S + wE(i)*rE(i);   K = K + wE(i)^2;
        else
            F(i) = false;  S = S - wE(i)*rE(i);   K = K - wE(i)^2;
        end
    end
    if isempty(nu)
        nu = S / (nw2 - K);
    end
end
