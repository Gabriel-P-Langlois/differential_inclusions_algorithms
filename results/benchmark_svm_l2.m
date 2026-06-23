%   BENCHMARK_SVM_L2    Synthetic L2-SVM (squared-hinge) benchmark for the
%                       differential-inclusions QP solvers.  (Script: all
%                       variables remain in the base workspace.)
%
%   We solve the bias-augmented squared-hinge SVM primal
%
%       min_{w,xi}  (1/2)||w||^2 + (C/2)||xi||^2
%       s.t.        y_i (w' x_i) + xi_i >= 1,   xi_i >= 0,   i = 1..n,
%
%   a convex QP with a DIAGONAL strictly positive-definite Hessian
%   Q = diag([ones(d,1); C*ones(n,1)]) (two scales, 1 and C; cond(Q) = C).
%   With z = (w, xi) and the bias folded into w (d = d_feat + 1), the DI form
%   of the constraints is A*z <= b with
%
%       A = [ -diag(y)*Xaug, -I_n ;  0, -I_n ],   b = [ -1 ; 0 ],
%
%   where Xaug = [X, 1].  There is no linear term, so after the diagonal
%   rescaling y = Q^{1/2} z the problem becomes min (1/2)||y||^2 s.t.
%   (A*Q^{-1/2}) y <= b, i.e. di_rlp's t = 1, c = 0 minimal-norm problem.
%   Hence the diagonal SVM inherits the FINITE-TERMINATION guarantee of the
%   Q = t*I theory: di_rlp solves it exactly in finitely many breakpoints,
%   which we count via the inner NNLS calls (the global SVM_NNLS_CALLS,
%   incremented by the local wrappers nnls_rlp / nnls_qp).
%
%   Solvers compared: di_rlp (finite termination), di_qp (general SPD,
%   descent), and quadprog / MOSEK / Gurobi / HiGHS as baselines.  The
%   hardness is active-set (support-vector) identification, not the cheap
%   diagonal inner solve.
%
%   The size sweep is a scale sweep at fixed sample/feature ratio and fixed
%   support-vector fraction (a single size d_feat = 49 by default; extend
%   d_feat_list for larger instances).

% Put the DI solvers (src/main) and NNLS solvers (src/utils/nnls) on path.
here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', '..', 'src')));

% Global NNLS-call counter, incremented by the local wrappers (script
% workaround for the nested-function counter, so this file stays a script).
global SVM_NNLS_CALLS


%% Parameters
rng(1);
d_feat_list = [2000 5000 10000];  % feature dimensions to sweep (single entry by default)
rho         = 2;         % samples per feature:  n = rho * d_feat
C_over_n    = 1;         % squared-hinge penalty as a multiple of n: C = C_over_n*n,
                         % i.e. regularization lambda = 1/(C_over_n*n).  The choice
                         % lambda ~ 1/n is the realistic low-regularization / near-
                         % separable regime in which the condition number L/lambda
                         % dominates first-order cost (Yang, Jin, Zhu & Lin 2015).
s_nnz       = 80;        % nonzeros per sample (capped at d_feat); denser samples
                         % couple more features -> stronger Gram collinearity
sv_target   = 0.25;      % target support-vector fraction (label-noise band)
k_factors   = 20;        % feature design: [] = iid near-orthogonal columns
                         % (well-conditioned Gram); integer k << d_feat = shared
                         % latent-factor design (correlated, ill-conditioned Gram --
                         % the conditioning knob; fewer factors = more collinear).
                         % Real data has low effective rank ~ O(log n) (Udell &
                         % Townsend 2019), so k is small and ~constant across sizes.
corr_noise  = 0.1;       % idiosyncratic noise in the latent-factor design
                         % (smaller = more collinear/degenerate; unused if k_factors=[])
tol         = 1e-8;      % outer tolerance for the DI solvers
warm_scale  = 1.0;       % norm of the random start direction w0 (0 = symmetric w=0 start)
warm_slack  = 0.5;       % interior gap delta in the feasible start (keeps it strictly feasible)

% Solver availability.
have_mosek  = exist('mosekopt',  'file') > 0;
have_gurobi = exist('gurobi',    'file') > 0;
have_highs  = exist('callhighs', 'file') > 0;

% Per-solver size caps: a solver is skipped when the problem size 4*d_feat
% (the number of inequality constraints, 2n) exceeds its cap.  Inf = always
% run.  HiGHS's active-set QP scales with the large SVM active set, so it is
% capped; raise/lower these freely.  di_rlp is never capped (it is the method
% under study and the reference fallback when quadprog is skipped).
diqp_N_max     = 0;
quadprog_N_max = 0;
mosek_N_max    = Inf;
gurobi_N_max   = Inf;
highs_N_max    = 0;

fprintf('\n=== Synthetic L2-SVM (squared-hinge) benchmark ===\n');
fprintf('C = %g*n  ratio n/d_feat = %g   sv_target = %.2f   k_factors = %s\n', ...
        C_over_n, rho, sv_target, mat2str(k_factors));
fprintf('externals:  MOSEK %d   Gurobi %d   HiGHS %d  (1 = available)\n', ...
        have_mosek, have_gurobi, have_highs);


%% Size sweep
for idx = 1:numel(d_feat_list)
    df = d_feat_list(idx);
    n  = rho * df;
    s  = min(s_nnz, df);              % nonzeros per (sparse) sample
    C  = C_over_n * n;                % penalty tied to n (lambda = 1/(C_over_n*n))
    d  = df + 1;                      % bias-augmented feature dimension
    nz = d + n;                       % total QP variables

    % --- Synthetic data ------------------------------------------------
    [X, y, w_star] = gen_svm(n, df, s, sv_target, k_factors, corr_noise);
    Xaug = [X, ones(n, 1)];           % bias as a constant feature
    Y    = spdiags(y, 0, n, n);

    % --- QP data -------------------------------------------------------
    Imarg = speye(n);
    A = [ -(Y * Xaug),    -Imarg ;     % margin:   -(y_i x_i' w + xi_i) <= -1
           sparse(n, d),  -Imarg ];    % box:      -xi_i <= 0
    b = [ -ones(n, 1); zeros(n, 1) ];
    qdiag = [ ones(d, 1); C * ones(n, 1) ];
    Qfun  = @(v) qdiag .* v;           % diagonal apply for di_qp
    Q     = spdiags(qdiag, 0, nz, nz); % explicit matrix for the externals
    fval  = @(z) 0.5 * (z' * (qdiag .* z));   % objective (no linear term)

    % Generic feasible start (avoids the slow Phase-1, deliberately NOT
    % near-optimal).  A uniform xi is infeasible for non-separable data
    % (misclassified points need xi_i > 1), so we fix a moderate random
    % direction w0 -- which also breaks the w = 0 symmetry that collapsed the
    % breakpoint path -- and set the slacks per sample so every margin holds
    % with interior gap warm_slack:  xi_i = max(0, 1 - y_i w0'x_i) + delta.
    w0  = warm_scale * randn(d, 1) / sqrt(d);   % O(1)-norm random direction
    m0  = (Y * Xaug) * w0;                       % signed margins y_i w0'x_i
    xi0 = max(0, 1 - m0) + warm_slack;           % feasible slacks, strictly interior
    z0  = [w0; xi0];

    fprintf('\n----- d_feat = %d   (n = %d,  vars = %d,  constraints = %d) -----\n', ...
            df, n, nz, 2 * n);

    % Per-solver run decisions: skip when the size 4*d_feat exceeds the cap.
    Ncap         = 4 * df;
    run_diqp     = (Ncap <= diqp_N_max);
    run_quadprog = (Ncap <= quadprog_N_max);
    run_mosek    = have_mosek  && (Ncap <= mosek_N_max);
    run_gurobi   = have_gurobi && (Ncap <= gurobi_N_max);
    run_highs    = have_highs  && (Ncap <= highs_N_max);

    % --- di_rlp (rescaled to t = 1; finite termination) ----------------
    sq   = sqrt(qdiag);
    Atil = A .* (1 ./ sq');            % column scaling: A * Q^{-1/2}
    y0   = sq .* z0;                   % rescaled feasible start (di_rlp solves in y)
    SVM_NNLS_CALLS = 0;
    tic;
    [yv, p_rlp, ~, flag_rlp] = di_rlp(Atil, b, zeros(nz, 1), 1, y0, tol, [], @nnls_rlp);
    t_rlp = toc;
    z_rlp  = yv ./ sq;                 % undo the rescaling
    it_rlp = SVM_NNLS_CALLS;

    % --- di_qp (general SPD via function handle; descent) --------------
    z_qp = []; t_qp = NaN; it_qp = NaN; flag_qp = NaN; p_qp = [];
    if run_diqp
        SVM_NNLS_CALLS = 0;
        tic;
        [z_qp, p_qp, ~, flag_qp] = di_qp(A, b, zeros(nz, 1), Qfun, z0, tol, [], @nnls_qp);
        t_qp  = toc;
        it_qp = SVM_NNLS_CALLS;
    else
        fprintf('  Skipping di_qp (4*d_feat = %d > %g).\n', Ncap, diqp_N_max);
    end

    % --- quadprog (independent reference; xi >= 0 as bounds, margin rows) --
    Aext = [ -(Y * Xaug), -Imarg ];
    bext = -ones(n, 1);
    lb   = [ -inf(d, 1); zeros(n, 1) ];
    z_quad = []; t_quad = NaN;
    if run_quadprog
        opts = optimoptions('quadprog', 'Display', 'none');
        tic;
        z_quad = quadprog(Q, zeros(nz, 1), Aext, bext, [], [], lb, [], [], opts);
        t_quad = toc;
    else
        fprintf('  Skipping quadprog (4*d_feat = %d > %g).\n', Ncap, quadprog_N_max);
    end

    % --- External baselines (best-effort, guarded) ---------------------
    z_mosek  = []; t_mosek  = NaN;
    z_gurobi = []; t_gurobi = NaN;
    z_highs  = []; t_highs  = NaN;

    if run_mosek
        try
            [qi, qj, qv] = find(tril(Q));
            prob.qosubi = qi; prob.qosubj = qj; prob.qoval = qv;
            prob.c   = zeros(nz, 1);
            prob.a   = Aext;
            prob.blc = -inf(n, 1);
            prob.buc = bext;
            prob.blx = lb;
            prob.bux = inf(nz, 1);
            tic;
            [~, res] = mosekopt('minimize echo(0)', prob, struct());
            t_mosek = toc;
            z_mosek = res.sol.itr.xx;
        catch err
            fprintf('  MOSEK failed: %s\n', err.message);
        end
    elseif have_mosek
        fprintf('  Skipping MOSEK (4*d_feat = %d > %g).\n', Ncap, mosek_N_max);
    end

    if run_gurobi
        try
            model.Q     = Q / 2;       % Gurobi omits the 1/2 factor
            model.obj   = zeros(nz, 1);
            model.A     = Aext;
            model.rhs   = bext;
            model.sense = repmat('<', n, 1);
            model.lb    = lb;
            model.ub    = inf(nz, 1);
            params.OutputFlag = 0;
            tic;
            rg = gurobi(model, params);
            t_gurobi = toc;
            z_gurobi = rg.x;
        catch err
            fprintf('  Gurobi failed: %s\n', err.message);
        end
    elseif have_gurobi
        fprintf('  Skipping Gurobi (4*d_feat = %d > %g).\n', Ncap, gurobi_N_max);
    end

    if run_highs
        try
            hopts.output_flag = false;
            tic;
            soln = callhighs(zeros(nz, 1), Aext, -inf(n, 1), bext, ...
                             lb, inf(nz, 1), Q, [], hopts);
            t_highs = toc;
            z_highs = soln.col_value;
        catch err
            fprintf('  HiGHS failed: %s\n', err.message);
        end
    elseif have_highs
        fprintf('  Skipping HiGHS (4*d_feat = %d > %g).\n', Ncap, highs_N_max);
    end

    % Reference for objgap/solgap: the independent quadprog solution when it
    % ran, else the exact di_rlp solution (machine-precision KKT).
    if ~isempty(z_quad)
        z_ref = z_quad;
    else
        z_ref = z_rlp;
    end
    f_ref = fval(z_ref);

    % --- Diagnostics (realism check) -----------------------------------
    xi_ref  = z_ref(d+1:end);
    sv_frac = mean(xi_ref > 1e-6);                       % active / violated margins
    w_sol   = z_ref(1:df);
    cosalign = (w_sol' * w_star) / (norm(w_sol) * norm(w_star) + realmin);
    fprintf('  support-vector fraction %.3f   cos(w, w_star) %.3f\n', sv_frac, cosalign);

    % --- Verification + table ------------------------------------------
    pfeas = @(z) max(0, max(A * z - b));
    solg  = @(z) norm(z - z_ref) / max(1, norm(z_ref));

    fprintf('\n  %-10s %10s %14s %12s %12s %12s %9s %6s\n', ...
            'solver', 'time(s)', 'fval', 'objgap', 'pfeas', 'solgap', 'iters', 'flag');
    print_row('di_rlp',   t_rlp,   fval(z_rlp), abs(fval(z_rlp)-f_ref), pfeas(z_rlp), solg(z_rlp), it_rlp, flag_rlp);
    if ~isempty(z_qp),   print_row('di_qp',    t_qp,   fval(z_qp),   abs(fval(z_qp)  -f_ref), pfeas(z_qp),   solg(z_qp),   it_qp, flag_qp); end
    if ~isempty(z_quad), print_row('quadprog', t_quad, fval(z_quad), abs(fval(z_quad)-f_ref), pfeas(z_quad), solg(z_quad), NaN,   NaN);     end
    if ~isempty(z_mosek),  print_row('mosek',  t_mosek,  fval(z_mosek),  abs(fval(z_mosek) -f_ref), pfeas(z_mosek),  solg(z_mosek),  NaN, NaN); end
    if ~isempty(z_gurobi), print_row('gurobi', t_gurobi, fval(z_gurobi), abs(fval(z_gurobi)-f_ref), pfeas(z_gurobi), solg(z_gurobi), NaN, NaN); end
    if ~isempty(z_highs),  print_row('highs',  t_highs,  fval(z_highs),  abs(fval(z_highs) -f_ref), pfeas(z_highs),  solg(z_highs),  NaN, NaN); end

    % Time ratios to the fastest available solver (single size).
    all_t = [t_rlp, t_qp, t_quad, t_mosek, t_gurobi, t_highs];
    fprintf('  fastest solve: %.4g s   di_rlp / fastest = %.2f x\n', ...
            min(all_t(~isnan(all_t))), t_rlp / min(all_t(~isnan(all_t))));

    % Full KKT triple for the DI solvers.  Duals transfer unchanged under the
    % di_rlp rescaling (Q z + A' p = 0 follows from y + Atil' p = 0), so p_rlp
    % satisfies the original-form stationarity; di_qp returns p in the original
    % (un-equilibrated) space.
    print_kkt('di_rlp KKT', z_rlp, p_rlp, qdiag, A, b);
    if ~isempty(z_qp)
        print_kkt('di_qp  KKT', z_qp, p_qp, qdiag, A, b);
    end
    fprintf('\n');
end


% ===================================================================== %
%  Local functions                                                      %
% ===================================================================== %

function [q, dvec] = nnls_rlp(M, rhs, tl, pw)
%   di_rlp NNLS wrapper (4-arg convention) that counts breakpoints.
global SVM_NNLS_CALLS
SVM_NNLS_CALLS = SVM_NNLS_CALLS + 1;
[q, dvec] = apgd_lsqnonneg(M, rhs, tl, pw);
end


function [q, dvec, vv] = nnls_qp(M, rhs, tl, pw, vw)
%   di_qp NNLS wrapper (5-arg convention) that counts outer iterations.
global SVM_NNLS_CALLS
SVM_NNLS_CALLS = SVM_NNLS_CALLS + 1;
[q, dvec, vv] = apgd_lsqnonneg(M, rhs, tl, pw, vw);
end


function [X, y, w_star] = gen_svm(n, df, s, sv_target, k_factors, corr_noise)
%   Sparse latent-linear SVM data with a label-noise band.  Each sample has
%   s nonzeros, is row-normalized to unit norm (standard SVM preprocessing,
%   and keeps the di_rlp margin rows O(1) since di_rlp does not equilibrate).
%
%   Feature design (controls Gram conditioning):
%     k_factors = []  -> iid N(0,1) entries on random support: columns are
%                        near-orthogonal, Gram well-conditioned (the easy case).
%     k_factors = k   -> shared latent-factor design.  Loadings L (df x k) and
%                        per-sample scores F (n x k) are drawn once; each nonzero
%                        entry is  X_ij = L(j,:) * F(i,:)' + corr_noise * noise.
%                        Features then co-vary through the k << df shared factors,
%                        so the Gram matrix is multicollinear / ill-conditioned;
%                        fewer factors and smaller corr_noise = more degenerate.
%                        Sparsity (s nonzeros per row) is preserved.
if nargin < 5, k_factors = []; end
if nargin < 6 || isempty(corr_noise), corr_noise = 0.1; end

w_star = randn(df, 1);
w_star = w_star / norm(w_star);

rows = repelem((1:n)', s);
cols = zeros(n * s, 1);
for i = 1:n
    cols((i-1)*s + (1:s)) = randperm(df, s);
end

if isempty(k_factors)
    vals = randn(n * s, 1);                       % iid near-orthogonal design
else
    L    = randn(df, k_factors);                  % feature loadings
    F    = randn(n,  k_factors);                  % per-sample factor scores
    vals = zeros(n * s, 1);
    for i = 1:n
        idx       = (i-1)*s + (1:s);
        c         = cols(idx);                    % this sample's support
        vals(idx) = L(c, :) * F(i, :)' + corr_noise * randn(s, 1);
    end
end
X = sparse(rows, cols, vals, n, df);

rn = sqrt(sum(X.^2, 2));
rn(rn == 0) = 1;
X = X ./ rn;

scores = X * w_star;
y = sign(scores);
y(y == 0) = 1;

% Flip the labels of the smallest-margin points (realistic boundary noise)
% to create a soft margin with roughly sv_target support vectors.
[~, ord] = sort(abs(scores), 'ascend');
nflip = round(sv_target * n);
y(ord(1:nflip)) = -y(ord(1:nflip));
end


function print_row(name, t, f, objgap, pf, sg, it, fl)
%   One results row.  '--' for inapplicable iteration / flag fields.
if isnan(it), its = '       --'; else, its = sprintf('%9d', it); end
if isnan(fl), fls = '    --';    else, fls = sprintf('%6d', fl); end
fprintf('  %-10s %10.4g %14.6e %12.2e %12.2e %12.2e %s %s\n', ...
        name, t, f, objgap, pf, sg, its, fls);
end


function print_kkt(label, z, p, qdiag, A, b)
%   Standard inequality-QP KKT triple (matching test_rlp.m):
%   stationarity ||Q z + A' p||_inf, dual feasibility max(0,-min p),
%   complementary slackness ||p .* (b - A z)||_inf.
stat = norm(qdiag .* z + A' * p, inf);
dfes = max(0, -min(p));
comp = norm(p .* (b - A * z), inf);
fprintf('  %-12s stat %.2e   dualfeas %.2e   compl %.2e\n', ...
        label, stat, dfes, comp);
end
