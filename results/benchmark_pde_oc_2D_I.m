%   BENCHMARK_PDE_OC_2D_I   2-D Poisson optimal control, state bounds only,
%                            consistent Q1 mass matrix, mesh sweep.
%
%   Reference: Pearson & Gondzio (2017), Numer. Math. 137:959-999.
%
%   Problem (continuous):
%
%       min_{y,u}  (1/2)*||y - yhat||^2_{L2(Omega)}
%                  + (beta/2)*||u||^2_{L2(Omega)}
%
%       s.t.       -Lap(y) = u   in  Omega = [0,1]^2,
%                  y = 0         on  dOmega,
%                  y_a <= y <= y_b.
%
%   Desired state (Pearson & Gondzio 2017, Sec. 5, Table 2):
%
%       yhat(x1,x2) = sin(pi*x1) * sin(pi*x2).
%
%   Discretization: Q1 finite elements on a uniform n x n grid,
%   h = 1/(n+1), N = n^2 interior DOFs.
%
%       K   = kron(K1D,M1D) + kron(M1D,K1D)    Q1 stiffness (N x N, SPD)
%       M   = kron(M1D,M1D)                     consistent Q1 mass (N x N, SPD)
%
%   NOTE: Both K and M are identical to the matrices assembled by IFISS in
%   the paper; the Kronecker product formulas are computationally equivalent
%   representations for uniform Cartesian grids.  K = K1D (x) M1D + M1D (x) K1D
%   reproduces the 9-point Q1 Laplacian stencil, and M = M1D (x) M1D is the
%   consistent (banded) Q1 mass.  This matches Pearson & Gondzio Sec. 5
%   exactly; no mass lumping is used.  The DI solvers below never invert M, so
%   the consistent mass costs only an extra sparse mat-vec relative to lumping.
%
%   NOTE: These MATLAB files were built with the assistance of Claude
%   (Anthropic) and may contain minor deviations from the paper.
%
%   Runs a mesh sweep over n in n_vals, comparing solvers:
%
%       (a) MOSEK (interior-point, reference)
%       (b) Gurobi (interior-point + active-set, commercial)
%       (c) HiGHS (interior-point, open-source)
%       (d) DAQP (active-set, dense; skipped for N > daqp_N_max)
%       (e) OSQP (ADMM, first-order)
%       (f) PIQP (proximal interior point, sparse)
%       (g) di_qp_eq with the default internal direct Cholesky of K^2+M^2
%       (h) di_qp_eq with K^2-preconditioned CG for (K^2+M^2)^{-1}, supplied
%           as a user PE_fun; K is prefactored once per mesh via sparse
%           Cholesky with AMD reordering.
%
%   Solvers (g) and (h) treat the full (y,u) QP through the equality +
%   inequality form of di_qp_eq.  The K^2-preconditioned system
%   K^{-1}(K^2+M^2)K^{-1} = I + (K^{-1}M)^2 has eigenvalues in
%   [1, 1 + 1/lambda_min(K,M)^2], where K*x = lambda*M*x is the discrete
%   Dirichlet eigenproblem for -Lap.  Since lambda_min(K,M) -> 2*pi^2 (the
%   first eigenvalue of -Lap on the unit square), the bound tends to
%
%       1 + 1/(2*pi^2)^2 = 1 + 1/(4*pi^4) ~ 1.0026.
%
%   Measured: exactly 4 CG iterations per PE_fun call, uniform in h (the
%   eigenvalue bound holds because M ~ h^2*I on the smooth modes that K^{-1}
%   leaves intact, while K^{-1} damps the oscillatory modes).
%
%   WARNING -- null-space reduction is deliberately EXCLUDED here.  One might
%   try to eliminate the control via u = M^{-1}*K*y (exact for K*y = M*u) and
%   solve the reduced box-constrained QP in the state y,
%
%       min_y  (1/2)*y'*Q_red*y + c_red'*y   s.t.  y_a <= y <= y_b,
%
%   with Q_red = M + beta*K*M^{-1}*K and c_red = -M*yd.  Do NOT do this for a
%   STATE-bounded problem.  Eliminating u leaves the *stiffening* operator
%   K^2, so cond(Q_red) ~ cond(K)^2 ~ h^{-4} (~6e4 at n = 36, growing without
%   bound under refinement).  A descent / projected-gradient method such as
%   di_qp works in the Euclidean metric, so its outer-iteration count scales
%   with cond(Q_red): di_qp needed ~4e4 outer steps (NNLS solves) at n = 36,
%   versus a handful for solvers (g) and (h).  The reduction is well
%   conditioned only when the eliminated variable leaves a *smoothing*
%   (compact) operator, which is the CONTROL-bounded case (eliminate y,
%   Q_red = M*K^{-1}*M*K^{-1}*M + beta*M, cond ~ 1, mesh independent; see
%   benchmark_pde_oc_3D_I).  For state bounds the right DI methods are (g) and
%   (h), which never form Q_red: they solve in the full (y,u) space and
%   project with a factorization of K^2+M^2 or the [1,1.0026]-preconditioned
%   CG above.
%
% -------------------------------------------------------------------------
%   FUNCTIONS CALLED
%       mosekopt         (MOSEK Optimization Toolbox, interior-point)
%       di_qp_eq         (equality + inequality form)
%
%   LOCAL FUNCTIONS
%       compute_kkt      compute KKT residuals for one solver (state bounds)
%       pe_cg_log        stateful PE_fun / z_proj using K^2-preconditioned CG
%       k_chol_solve     apply K^{-1} via stored sparse Cholesky factor
%       print_row        print one solver row in the summary table
%       fmtsgm           format an SGM ratio for the ranking table
%
% -------------------------------------------------------------------------

%% Paths

script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', '..', 'src', 'main'));
addpath(fullfile(script_dir, '..', '..', 'src', 'utils', 'nnls'));

p_gurobi = '/Library/gurobi1302/macos_universal2/matlab';
p_highs  = fullfile(getenv('HOME'), 'HiGHSMEX');
p_osqp   = fullfile(getenv('HOME'), 'osqp');
p_daqp   = fullfile(getenv('HOME'), 'daqp-matlab', 'daqp-matlab-mac64');
p_piqp   = fullfile(getenv('HOME'), 'piqp-matlab-maca64');

if isfolder(p_gurobi), addpath(p_gurobi); end
if isfolder(p_highs),  addpath(p_highs);  end
if isfolder(p_osqp),   addpath(p_osqp);   end
if isfolder(p_daqp),   addpath(p_daqp);   end
if isfolder(p_piqp),   addpath(p_piqp);   end


%% Parameters

n_vals = [4, 16, 32, 64];   % interior grid points per dimension

% Skip if > than the numbers below
mosek_N_max    = inf;
gurobi_N_max   = inf;
highs_N_max    = 128^2;
daqp_N_max     = 8^2;
osqp_N_max     = 16^2;
piqp_N_max     = 36^2;
chol_N_max     = Inf;    % direct Chol of K^2+M^2 is affordable in 2D

% state regularization and bounds (Pearson & Gondzio 2017, Table 2, beta=1e-2)
beta = 1e-2;
y_a  = -0.1;    y_b = 0.175;

pcg_tol   = 1e-12;   % CG convergence tolerance inside PE_fun
pcg_maxit = 20;      % CG iteration cap (theory predicts ~1 on fine meshes)

tol_kkt   = 1e-08;
tol_di_eq = 1e-08;

fprintf('Benchmark: 2-D Poisson OC, state bounds only, consistent Q1 mass, mesh sweep.\n')
fprintf('  beta = %g,  y in [%g, %g]\n', beta, y_a, y_b)
fprintf('  Desired state: sin(pi*x1)*sin(pi*x2).\n')
fprintf('  CG settings: tol = %g,  maxit = %d\n\n', pcg_tol, pcg_maxit)


%% Solver availability

have_mosek  = exist('mosekopt',  'file')  > 0;
have_gurobi = exist('gurobi',    'file')  > 0;
have_highs  = exist('callhighs', 'file')  > 0;
have_daqp   = exist('daqp',      'class') > 0;
have_osqp   = exist('osqp',      'class') > 0;
have_piqp   = exist('piqp',      'file')  > 0;

avail_rows = { ...
    'MOSEK',   have_mosek,  mosek_N_max;  ...
    'Gurobi',  have_gurobi, gurobi_N_max; ...
    'HiGHS',   have_highs,  highs_N_max;  ...
    'DAQP',    have_daqp,   daqp_N_max;   ...
    'OSQP',    have_osqp,   osqp_N_max;   ...
    'PIQP',    have_piqp,   piqp_N_max;   ...
    'Chol',    true,        chol_N_max;   ...
    'CGeq',    true,        Inf           };

fprintf('Solver availability:\n')
fprintf('  %-10s  %-12s  %s\n', 'Solver', 'Status', 'Size limit')
fprintf('  %s\n', repmat('-', 1, 42))
for k = 1:size(avail_rows, 1)
    name  = avail_rows{k,1};
    avail = avail_rows{k,2};
    nlim  = avail_rows{k,3};
    if avail,    stat = 'found';
    else,        stat = 'NOT FOUND'; end
    if isinf(nlim), lim_str = 'no limit';
    else,           lim_str = sprintf('N <= %g', nlim); end
    fprintf('  %-10s  %-12s  %s\n', name, stat, lim_str)
end
fprintf('\n')


%% Result storage

nn = length(n_vals);
pfx = {'mosek','gurobi','highs','daqp','osqp','piqp','chol','cgeq'};
res = struct('n', zeros(nn,1), 'N', zeros(nn,1));
for k = 1:numel(pfx)
    p = pfx{k};
    res.(['t_'         p]) = nan(nn,1);
    res.(['fval_'      p]) = nan(nn,1);
    res.(['kkt_stat_'  p]) = nan(nn,1);
    res.(['kkt_pfeas_' p]) = nan(nn,1);
    res.(['kkt_dfeas_' p]) = nan(nn,1);
    res.(['fgap_'      p]) = nan(nn,1);
end
res.cgeq_calls = zeros(nn,1);
res.cgeq_iters = zeros(nn,1);


%% Loop over mesh sizes

for idx = 1:nn

    n = n_vals(idx);
    N = n^2;
    h = 1/(n+1);

    run_mosek  = have_mosek  && (N <= mosek_N_max);
    run_gurobi = have_gurobi && (N <= gurobi_N_max);
    run_highs  = have_highs  && (N <= highs_N_max);
    run_daqp   = have_daqp   && (N <= daqp_N_max);
    run_osqp   = have_osqp   && (N <= osqp_N_max);
    run_piqp   = have_piqp   && (N <= piqp_N_max);
    run_chol   = (N <= chol_N_max);

    fprintf('=== n = %d  (N = %d DOFs,  h = 1/%d) ===\n\n', n, N, n+1)


    %% Grid and desired state

    x1vec    = (1:n)' * h;
    [X1, X2] = ndgrid(x1vec, x1vec);
    Yhat     = sin(pi*X1) .* sin(pi*X2);
    yd       = Yhat(:);


    %% FEM matrices: consistent Q1 stiffness and mass (IFISS)

    e   = ones(n, 1);
    K1D = spdiags([-e, 2*e, -e], [-1,0,1], n, n) / h;
    M1D = spdiags([ e, 4*e,  e], [-1,0,1], n, n) * (h/6);

    K     = kron(K1D, M1D) + kron(M1D, K1D);   % N x N, SPD
    M_mat = kron(M1D, M1D);                       % consistent Q1 mass


    %% QP data (full (y,u) formulation, state bounds)

    H    = blkdiag(M_mat, beta * M_mat);
    fvec = [-M_mat * yd; sparse(N, 1)];
    Aeq  = [K, -M_mat];
    beq  = zeros(N, 1);
    lb   = [y_a * ones(N, 1);  -inf(N, 1)];
    ub   = [y_b * ones(N, 1);   inf(N, 1)];


    %% Constraint matrices for di_qp_eq (state bounds only)

    E_eq = [K, -M_mat];                          % N x 2N

    % Row blocks (each N rows): y <= y_b, y >= y_a.
    A_di_eq = [ speye(N),  sparse(N,N); ...
               -speye(N),  sparse(N,N)];          % 2N x 2N
    b_di_eq = [y_b*ones(N,1); -y_a*ones(N,1)];


    %% Feasible starting point: y0 = 0, u0 = 0 (K*0 = M*0, y_a <= 0 <= y_b)

    x0 = zeros(2*N, 1);


    % KKT diagnostics reported for each solver (STATE-bounded problem):
    %
    %   kkt_stat  = ||r_y||_inf over components of y strictly interior to
    %               (y_a, y_b), where r_y = M*(y - yd) + K*mu is the state
    %               stationarity residual.  Zero at an exact KKT point.
    %   kkt_pfeas = max(||K*y - M*u||_inf,  max(0, max([y-y_b; y_a-y]))).
    %               Primal feasibility: max of the PDE equality residual and
    %               the state-bound violation.
    %   kkt_dfeas = max of: max(0, r_y) over components at y_b (r_y <= 0
    %               there), max(0, -r_y) over components at y_a (r_y >= 0
    %               there), and -- only for full_diag solvers -- the
    %               free-control stationarity ||mu - beta*u||_inf (zero by
    %               construction for the DI solvers).


    %% 1. MOSEK

    if run_mosek
        tic
        [qi, qj, qv] = find(tril(H));
        prob_mosek.qosubi = qi;
        prob_mosek.qosubj = qj;
        prob_mosek.qoval  = qv;
        prob_mosek.c      = full(fvec);
        prob_mosek.a      = Aeq;
        prob_mosek.blc    = beq;
        prob_mosek.buc    = beq;
        prob_mosek.blx    = lb;
        prob_mosek.bux    = ub;
        param.MSK_IPAR_LOG = 0;

        fprintf('  Running MOSEK...\n')
        [~, res_mosek] = mosekopt('minimize echo(0)', prob_mosek, param);
        t_mosek = toc;

        x_mosek      = res_mosek.sol.itr.xx;
        fval_mosek   = res_mosek.sol.itr.pobjval;
        flag_mosek   = res_mosek.rcode;
        mu_mosek     = -res_mosek.sol.itr.y;   % MOSEK uses opposite sign convention

        y_mosek = x_mosek(1:N);
        u_mosek = x_mosek(N+1:2*N);

        fgap_mosek = res_mosek.sol.itr.pobjval - res_mosek.sol.itr.dobjval;
        [ks, kp, kd] = compute_kkt(y_mosek, u_mosek, mu_mosek, K, M_mat, yd, beta, y_a, y_b, tol_kkt, true);

        fprintf('  MOSEK:  t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (flag=%d)\n\n', ...
                t_mosek, fval_mosek, ks, kp, kd, fgap_mosek, flag_mosek)

        res = store(res, 'mosek', idx, t_mosek, fval_mosek, ks, kp, kd, fgap_mosek);
    else
        skip_msg('MOSEK', have_mosek, N, mosek_N_max)
    end


    %% 2. Gurobi (interior-point)
    %   Gurobi minimizes x'*Q*x + obj'*x (no 1/2 factor), so pass Q = H/2.
    %   result.pi is the equality adjoint mu directly (no sign flip).
    %   rc = Hx+f+Aeq'*mu = nu_lb - nu_ub at KKT; only the y-block is bounded.

    if run_gurobi
        tic
        model_g.Q     = H / 2;
        model_g.obj   = full(fvec);
        model_g.A     = Aeq;
        model_g.rhs   = full(beq);
        model_g.sense = repmat('=', N, 1);
        model_g.lb    = full(lb);
        model_g.ub    = full(ub);
        params_g.OutputFlag = 0;

        fprintf('  Running Gurobi...\n')
        result_g = gurobi(model_g, params_g);
        rc_y_g = result_g.rc(1:N);
        fgap_gurobi = result_g.pi' * (beq - Aeq * result_g.x) ...
                    + max(0,  rc_y_g)' * (result_g.x(1:N) - y_a) ...
                    + max(0, -rc_y_g)' * (y_b - result_g.x(1:N));
        t_gurobi = toc;

        if ~strcmp(result_g.status, 'OPTIMAL')
            error('Gurobi did not find an optimal solution (status: %s).', result_g.status)
        end

        y_gurobi = result_g.x(1:N);
        u_gurobi = result_g.x(N+1:2*N);
        [ks, kp, kd] = compute_kkt(y_gurobi, u_gurobi, result_g.pi, K, M_mat, yd, beta, y_a, y_b, tol_kkt, true);

        fprintf('  Gurobi: t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (status=%s)\n\n', ...
                t_gurobi, result_g.objval, ks, kp, kd, fgap_gurobi, result_g.status)

        res = store(res, 'gurobi', idx, t_gurobi, result_g.objval, ks, kp, kd, fgap_gurobi);
    else
        skip_msg('Gurobi', have_gurobi, N, gurobi_N_max)
    end


    %% 3. HiGHS (interior-point)
    %   HiGHS uses (1/2)x'Qx + c'x; pass H directly.  Adjoint mu = -row_dual.
    %   col_dual = nu_lb - nu_ub at KKT; only the y-block is bounded.

    if run_highs
        tic
        opts_h.output_flag                  = false;
        opts_h.solver                       = "ipm";
        opts_h.ipm_optimality_tolerance     = 1e-10;
        opts_h.primal_feasibility_tolerance = 1e-10;
        opts_h.dual_feasibility_tolerance   = 1e-10;

        fprintf('  Running HiGHS (IPM)...\n')
        [soln_h, info_h] = callhighs(full(fvec), Aeq, beq, beq, lb, ub, H, [], opts_h);
        rc_y_h = soln_h.col_dual(1:N);
        mu_h   = -soln_h.row_dual;
        fgap_highs = mu_h' * (beq - Aeq * soln_h.col_value) ...
                   + max(0,  rc_y_h)' * (soln_h.col_value(1:N) - y_a) ...
                   + max(0, -rc_y_h)' * (y_b - soln_h.col_value(1:N));
        t_highs = toc;

        if ~strcmp(info_h.model_status_string, 'Optimal')
            error('HiGHS did not find an optimal solution (status: %s).', info_h.model_status_string)
        end

        y_highs = soln_h.col_value(1:N);
        u_highs = soln_h.col_value(N+1:2*N);
        [ks, kp, kd] = compute_kkt(y_highs, u_highs, mu_h, K, M_mat, yd, beta, y_a, y_b, tol_kkt, true);

        fprintf('  HiGHS:  t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (status=%s)\n\n', ...
                t_highs, info_h.objective_function_value, ks, kp, kd, fgap_highs, info_h.model_status_string)

        res = store(res, 'highs', idx, t_highs, info_h.objective_function_value, ks, kp, kd, fgap_highs);
    else
        skip_msg('HiGHS', have_highs, N, highs_N_max)
    end


    %% 4. DAQP (active-set, dense)
    %   Simple bounds (ms = 2N): y in [y_a, y_b], u in (-Inf, Inf).
    %   Equality K*y = M*u as a general constraint (blower = bupper = beq).
    %   Standard KKT dual sign (no flip): lambda(1:N) = y-bound duals,
    %   lambda(2N+1:3N) = equality adjoint mu.

    if run_daqp
        tic
        ms_d     = 2*N;
        bupper_d = [y_b*ones(N,1);  inf(N,1);  full(beq)];
        blower_d = [y_a*ones(N,1); -inf(N,1);  full(beq)];
        sense_d  = zeros(ms_d + N, 1, 'int32');

        fprintf('  Running DAQP (active-set)...\n')
        [x_d, fval_d, exitflag_d, info_d] = daqp.quadprog( ...
            full(H), full(fvec), full(Aeq), bupper_d, blower_d, sense_d);
        t_daqp = toc;

        if exitflag_d ~= int32(1)
            warning('DAQP did not find an optimal solution (exitflag: %d).', exitflag_d)
        end

        lambda_d   = info_d.lambda;
        mu_eq_d    = lambda_d(ms_d+1:ms_d+N);   % equality adjoint (standard KKT sign)
        lambda_y_d = lambda_d(1:N);              % state bound multipliers

        fgap_daqp = mu_eq_d' * (beq - Aeq * x_d) ...
                  + max(0, -lambda_y_d)' * (x_d(1:N) - y_a) ...
                  + max(0,  lambda_y_d)' * (y_b - x_d(1:N));

        y_d = x_d(1:N);
        u_d = x_d(N+1:2*N);
        [ks, kp, kd] = compute_kkt(y_d, u_d, mu_eq_d, K, M_mat, yd, beta, y_a, y_b, tol_kkt, true);

        fprintf('  DAQP:   t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (exitflag=%d)\n\n', ...
                t_daqp, fval_d, ks, kp, kd, fgap_daqp, exitflag_d)

        res = store(res, 'daqp', idx, t_daqp, fval_d, ks, kp, kd, fgap_daqp);
    else
        skip_msg('DAQP', have_daqp, N, daqp_N_max)
    end


    %% 5. OSQP (ADMM, first-order)
    %   Bounds encoded as rows: A_osqp = [Aeq; [I, 0]].  Standard KKT sign
    %   (no flip): y(1:N) equality adjoint, y(N+1:2N) state-bound dual
    %   (>= 0 at y = y_b, <= 0 at y = y_a).

    if run_osqp
        tic
        A_osqp  = [Aeq; speye(N), sparse(N,N)];
        l_osqp  = [beq;    y_a * ones(N,1)];
        ub_osqp = [beq;    y_b * ones(N,1)];

        prob_osqp = osqp;
        prob_osqp.setup(triu(H), full(fvec), A_osqp, l_osqp, ub_osqp, ...
            'verbose',  false, 'eps_abs', 1e-06, 'eps_rel', 1e-06, ...
            'polish',   true,  'max_iter', 10000);

        fprintf('  Running OSQP (ADMM)...\n')
        res_osqp = prob_osqp.solve();
        t_osqp = toc;

        if res_osqp.info.status_val ~= 1
            warning('OSQP did not find an optimal solution (status: %s).', res_osqp.info.status)
        end

        mu_eq_osqp = res_osqp.y(1:N);
        y_y_osqp   = res_osqp.y(N+1:2*N);
        fgap_osqp = mu_eq_osqp' * (beq - Aeq * res_osqp.x) ...
                  + max(0, -y_y_osqp)' * (res_osqp.x(1:N) - y_a) ...
                  + max(0,  y_y_osqp)' * (y_b - res_osqp.x(1:N));

        y_osqp = res_osqp.x(1:N);
        u_osqp = res_osqp.x(N+1:2*N);
        [ks, kp, kd] = compute_kkt(y_osqp, u_osqp, mu_eq_osqp, K, M_mat, yd, beta, y_a, y_b, tol_kkt, true);

        fprintf('  OSQP:   t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (status=%s)\n\n', ...
                t_osqp, res_osqp.info.obj_val, ks, kp, kd, fgap_osqp, res_osqp.info.status)

        res = store(res, 'osqp', idx, t_osqp, res_osqp.info.obj_val, ks, kp, kd, fgap_osqp);
    else
        skip_msg('OSQP', have_osqp, N, osqp_N_max)
    end


    %% 6. PIQP (proximal interior point)
    %   Box bounds lb <= x <= ub passed directly; no general inequalities.
    %   Standard KKT sign: result.y equality adjoint; z_bl, z_bu box duals
    %   (>= 0).  Bounded block is y (1:N).

    if run_piqp
        tic
        G_p   = sparse(0, 2*N);
        h_l_p = zeros(0, 1);
        h_u_p = zeros(0, 1);

        solver_p = piqp('sparse');
        solver_p.update_settings('verbose', false, 'eps_abs', 1e-9, ...
            'eps_rel', 1e-9, 'compute_timings', true);
        solver_p.setup(H, full(fvec), Aeq, full(beq), G_p, h_l_p, h_u_p, full(lb), full(ub));

        fprintf('  Running PIQP (proximal IPM)...\n')
        result_p = solver_p.solve();
        t_piqp = toc;

        if ~strcmp(result_p.info.status, 'solved')
            warning('PIQP did not find an optimal solution (status: %s).', result_p.info.status)
        end

        mu_eq_piqp = result_p.y;
        z_bl_y     = result_p.z_bl(1:N);
        z_bu_y     = result_p.z_bu(1:N);
        fgap_piqp = mu_eq_piqp' * (beq - Aeq * result_p.x) ...
                  + z_bl_y' * (result_p.x(1:N) - y_a) ...
                  + z_bu_y' * (y_b - result_p.x(1:N));

        y_piqp = result_p.x(1:N);
        u_piqp = result_p.x(N+1:2*N);
        [ks, kp, kd] = compute_kkt(y_piqp, u_piqp, mu_eq_piqp, K, M_mat, yd, beta, y_a, y_b, tol_kkt, true);

        fprintf('  PIQP:   t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (status=%s)\n\n', ...
                t_piqp, result_p.info.primal_obj, ks, kp, kd, fgap_piqp, result_p.info.status)

        res = store(res, 'piqp', idx, t_piqp, result_p.info.primal_obj, ks, kp, kd, fgap_piqp);
    else
        skip_msg('PIQP', have_piqp, N, piqp_N_max)
    end


    %% 7. di_qp_eq, direct Cholesky of K^2+M^2

    if run_chol
        fprintf('  Running di_qp_eq (direct Cholesky)...\n')
        tic
        [x_chol, ~, mu_chol, fval_chol, flag_chol, fgap_chol] = ...
            di_qp_eq(A_di_eq, b_di_eq, fvec, H, E_eq, beq, x0, tol_di_eq);
        t_chol = toc;

        y_chol = x_chol(1:N);
        u_chol = x_chol(N+1:2*N);
        [ks, kp, kd] = compute_kkt(y_chol, u_chol, mu_chol, K, M_mat, yd, beta, y_a, y_b, tol_kkt, false);

        fprintf('  Chol:   t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (flag=%d)\n\n', ...
                t_chol, fval_chol, ks, kp, kd, fgap_chol, flag_chol)

        res = store(res, 'chol', idx, t_chol, fval_chol, ks, kp, kd, fgap_chol);
    else
        fprintf('  Skipping direct Cholesky (N = %d > %g).\n\n', N, chol_N_max)
    end


    %% Prefactor K once (used by solver 8)
    %   K(perm_K, perm_K) = L_K * L_K'  (AMD reordering).

    [L_K, ~, perm_K] = chol(K, 'lower', 'vector');


    %% 8. di_qp_eq with K^2-preconditioned CG PE_fun

    pe_cg_log('reset');
    PE_fun_cg = @(v) pe_cg_log('apply', K, M_mat, E_eq, L_K, perm_K, pcg_tol, pcg_maxit, v);

    A_op_eet     = @(w) K*(K*w) + M_mat*(M_mat*w);
    prec_eet     = @(r) k_chol_solve(L_K, perm_K, k_chol_solve(L_K, perm_K, r));
    EEt_solve_cg = @(r) pcg(A_op_eet, r, pcg_tol, pcg_maxit, prec_eet);

    fprintf('  Running di_qp_eq (K^2-preconditioned CG)...\n')
    tic
    [x_cg, ~, mu_cg, fval_cg, flag_cg, fgap_cg] = ...
        di_qp_eq(A_di_eq, b_di_eq, fvec, H, E_eq, beq, x0, tol_di_eq, [], PE_fun_cg, EEt_solve_cg);
    t_cgeq = toc;

    [cg_total_iters, cg_num_calls] = pe_cg_log('get');

    y_cg = x_cg(1:N);
    u_cg = x_cg(N+1:2*N);
    [ks, kp, kd] = compute_kkt(y_cg, u_cg, mu_cg, K, M_mat, yd, beta, y_a, y_b, tol_kkt, false);

    fprintf('  CGeq:   t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (flag=%d)\n', ...
            t_cgeq, fval_cg, ks, kp, kd, fgap_cg, flag_cg)
    fprintf('          CG calls = %d,  total iters = %d,  avg = %.2f iters/call\n\n', ...
            cg_num_calls, cg_total_iters, cg_total_iters / max(cg_num_calls, 1))

    res = store(res, 'cgeq', idx, t_cgeq, fval_cg, ks, kp, kd, fgap_cg);
    res.cgeq_calls(idx) = cg_num_calls;
    res.cgeq_iters(idx) = cg_total_iters;


    %% Store mesh sizes

    res.n(idx) = n;
    res.N(idx) = N;

end  % mesh loop


%% Build data matrices for per-mesh SGM rankings

sgm_solver_names = {'Chol (di_qp_eq)','CGeq (di_qp_eq)','MOSEK', ...
                    'Gurobi','HiGHS','OSQP','PIQP','DAQP'};
sgm_field_pfx    = {'chol','cgeq','mosek','gurobi','highs','osqp','piqp','daqp'};
ns_sgm = numel(sgm_solver_names);

T_sgm = nan(ns_sgm, nn);
P_sgm = nan(ns_sgm, nn);
D_sgm = nan(ns_sgm, nn);
S_sgm = nan(ns_sgm, nn);
G_sgm = nan(ns_sgm, nn);
for j = 1 : ns_sgm
    pf = sgm_field_pfx{j};
    tv = res.(['t_' pf])';
    ok = ~isnan(tv);
    T_sgm(j, ok) = tv(ok);
    P_sgm(j, ok) = res.(['kkt_pfeas_' pf])(ok)';
    D_sgm(j, ok) = res.(['kkt_dfeas_' pf])(ok)';
    S_sgm(j, ok) = res.(['kkt_stat_'  pf])(ok)';
    G_sgm(j, ok) = abs(res.(['fgap_'    pf])(ok)');
end


%% Summary table

hdr = sprintf('  %-20s  %11s  %12s  %12s  %12s  %12s', ...
    'Solver', 'Runtime (s)', 'Primal res', 'Dual res', 'Stat. compl.', 'Duality gap');
sep = ['  ' repmat('-', 1, length(hdr) - 2)];

print_pfx   = {'chol','cgeq','mosek','gurobi','highs','osqp','piqp','daqp'};
print_names = {'Chol (di_qp_eq)','CGeq (di_qp_eq)','MOSEK', ...
               'Gurobi','HiGHS','OSQP','PIQP','DAQP'};

fprintf('\nSummary\n')

for idx = 1:nn
    fprintf('\n=== n = %d,  N = %d,  h = 1/%d ===\n\n', ...
            res.n(idx), res.N(idx), res.n(idx) + 1)
    fprintf('%s\n', hdr)
    fprintf('%s\n', sep)

    for j = 1 : numel(print_pfx)
        p = print_pfx{j};
        if ~isnan(res.(['t_' p])(idx))
            print_row(print_names{j}, res.(['t_' p])(idx), ...
                [res.(['kkt_pfeas_' p])(idx), res.(['kkt_dfeas_' p])(idx), ...
                 res.(['kkt_stat_'  p])(idx), res.(['fgap_' p])(idx)])
        else
            fprintf('  %-20s  %11s  %12s  %12s  %12s  %12s\n', ...
                    [print_names{j} ' (skipped)'], '---', '---', '---', '---', '---')
        end
    end

    if res.cgeq_calls(idx) > 0
        fprintf('  (CGeq: %d CG calls, %.2f iters/call avg)\n', ...
            res.cgeq_calls(idx), res.cgeq_iters(idx) / res.cgeq_calls(idx))
    end

    % --- Per-mesh SGM ranking (ratios relative to best solver at this n) ---
    tc = T_sgm(:, idx);  pc = P_sgm(:, idx);  dc = D_sgm(:, idx);
    sc = S_sgm(:, idx);  gc = G_sgm(:, idx);

    rt = tc ./ min(tc(~isnan(tc)));
    rp = pc ./ min(pc(~isnan(pc)));
    rd = dc ./ min(dc(~isnan(dc)));
    rs = sc ./ min(sc(~isnan(sc)));
    rg = gc ./ min(gc(~isnan(gc)));

    hdr_sgm = sprintf('  %-22s  %12s  %12s  %12s  %12s  %12s', ...
        'Solver', 'Runtime', 'Primal res', 'Dual res', 'Stat. compl.', 'Duality gap');
    fprintf('\n  SGM ratios (relative to best at n = %d):\n', res.n(idx))
    fprintf('%s\n  %s\n', hdr_sgm, repmat('-', 1, length(hdr_sgm) - 2))
    for j = 1 : ns_sgm
        fprintf('  %-22s  %s  %s  %s  %s  %s\n', sgm_solver_names{j}, ...
            fmtsgm(rt(j)), fmtsgm(rp(j)), fmtsgm(rd(j)), ...
            fmtsgm(rs(j)), fmtsgm(rg(j)))
    end
    fprintf('\n\n')
end


%% =========================================================================
%  Local functions
%  =========================================================================

function res = store(res, pfx, idx, t, fval, kstat, kpfeas, kdfeas, fgap)
%STORE  Write one solver's results into the res struct for mesh index idx.
    res.(['t_'         pfx])(idx) = t;
    res.(['fval_'      pfx])(idx) = fval;
    res.(['kkt_stat_'  pfx])(idx) = kstat;
    res.(['kkt_pfeas_' pfx])(idx) = kpfeas;
    res.(['kkt_dfeas_' pfx])(idx) = kdfeas;
    res.(['fgap_'      pfx])(idx) = fgap;
end

function skip_msg(name, have, N, nlim)
%SKIP_MSG  Print a uniform skip message for an unavailable / oversized solver.
    if ~have
        fprintf('  Skipping %s (not found).\n\n', name)
    else
        fprintf('  Skipping %s (N = %d > %g).\n\n', name, N, nlim)
    end
end

function [kkt_stat, kkt_pfeas, kkt_dfeas] = ...
        compute_kkt(y, u, mu, K, M_mat, yd, beta, y_a, y_b, tol_kkt, full_diag)
%COMPUTE_KKT  KKT residuals for the 2-D Poisson OC problem (state bounds).
%
%   Inputs
%       y, u        primal state and control
%       mu          equality dual (adjoint / co-state)
%       full_diag   true: include the free-control stationarity
%                   ||mu - beta*u||_inf in kkt_dfeas (for solvers that do not
%                   enforce it by construction, e.g. MOSEK).  false: omit
%                   (zero by construction for the DI solvers).
%
%   Outputs
%       kkt_stat    ||r_y||_inf over interior states, r_y = M*(y-yd) + K*mu
%       kkt_pfeas   max(||K*y - M*u||_inf, state-bound violation)
%       kkt_dfeas   max of multiplier sign violations (+ control stat. if full_diag)

    r_y      = M_mat*(y - yd) + K*mu;
    interior = (y > y_a + tol_kkt) & (y < y_b - tol_kkt);
    if any(interior)
        kkt_stat = norm(r_y(interior), inf);
    else
        kkt_stat = 0;
    end

    eq_res   = norm(K*y - M_mat*u, inf);
    ineq_res = max(0, max([y - y_b; y_a - y]));
    kkt_pfeas = max(eq_res, ineq_res);

    at_ub = y >= y_b - tol_kkt;   % r_y <= 0 expected here
    at_lb = y <= y_a + tol_kkt;   % r_y >= 0 expected here
    kkt_dfeas = max(max([0; r_y(at_ub)]), max([0; -r_y(at_lb)]));

    if full_diag
        ctrl_stat = norm(mu - beta*u, inf);
        kkt_dfeas = max(kkt_dfeas, ctrl_stat);
    end
end

function varargout = pe_cg_log(cmd, varargin)
%PE_CG_LOG  Stateful PE_fun using K^2-preconditioned CG.
%
%   pe_cg_log('reset')                       reset the CG iteration counters.
%   [total_iters, num_calls] = pe_cg_log('get')   retrieve accumulated counts.
%   Pv = pe_cg_log('apply', K, M, E_eq, L_K, perm_K, pcg_tol, pcg_maxit, v)
%       Apply P_E*v = v - E_eq'*(K^2+M^2)^{-1}*(E_eq*v), using
%       K^2-preconditioned CG to solve (K^2+M^2)*w = E_eq*v, where
%       (K^2+M^2)*w = K*(K*w) + M*(M*w).  Preconditioner K^{-2} (two triangular
%       solves with L_K per CG step).  K(perm_K, perm_K) = L_K * L_K'.

    persistent total_iters num_calls
    if isempty(total_iters), total_iters = 0; num_calls = 0; end

    varargout = cell(1, nargout);

    switch cmd
        case 'reset'
            total_iters = 0;
            num_calls   = 0;

        case 'get'
            varargout{1} = total_iters;
            varargout{2} = num_calls;

        case 'apply'
            [K, M, E_eq, L_K, perm_K, pcg_tol, pcg_maxit, v] = deal(varargin{:});

            Ev   = E_eq * v;
            A_op = @(w) K*(K*w) + M*(M*w);
            prec = @(r) k_chol_solve(L_K, perm_K, ...
                            k_chol_solve(L_K, perm_K, r));

            warning('off', 'MATLAB:pcg:tooSmallTolerance')
            [w, ~, ~, iter] = pcg(A_op, Ev, pcg_tol, pcg_maxit, prec);
            warning('on',  'MATLAB:pcg:tooSmallTolerance')

            total_iters = total_iters + iter;
            num_calls   = num_calls   + 1;

            varargout{1} = v - E_eq' * w;

        otherwise
            error('pe_cg_log: unknown command ''%s''.', cmd)
    end
end

function x = k_chol_solve(L, perm, r)
%K_CHOL_SOLVE  Apply K^{-1} via the stored sparse Cholesky factor.
%   K(perm, perm) = L * L'  (lower triangular, AMD-ordered).
    x       = zeros(length(r), 1);
    x(perm) = L' \ (L \ r(perm));
end

function str = fmtsgm(v)
%FMTSGM  Format an SGM ratio as a 12-char string, or '---' for NaN.
    if isnan(v), str = sprintf('%12s', '---');
    else,        str = sprintf('%12.3e', v); end
end

function print_row(name, t, vals)
%PRINT_ROW  Print one solver row in the per-n summary table.
%   vals = [primal_res, dual_res, stat_compl, duality_gap].
    fprintf('  %-20s  %11.3f  %12.2e  %12.2e  %12.2e  %12.2e\n', ...
            name, t, vals(1), vals(2), vals(3), vals(4));
end
