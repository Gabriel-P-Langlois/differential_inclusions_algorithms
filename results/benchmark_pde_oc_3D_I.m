%   BENCHMARK_PDE_OC_3D_I   3-D Poisson optimal control, control bounds 
%                             only, consistent Q1 mass matrix, mesh sweep.
%
%   Reference: Pearson & Gondzio (2017), Numer. Math. 137:959-999.
%
%   Problem (continuous):
%
%       min_{y,u}  (1/2)*||y - yhat||^2_{L2(Omega)}
%                  + (beta/2)*||u||^2_{L2(Omega)}
%
%       s.t.       -Lap(y) = u   in  Omega = [0,1]^3,
%                  y = 0         on  dOmega,
%                  u_a <= u <= u_b.
%
%   Desired state (Pearson & Gondzio 2017, Sec. 5, Table 6):
%
%       yhat(x1,x2,x3) = exp(-64*((x1-0.5)^2 + (x2-0.5)^2 + (x3-0.5)^2)).
%
%   Discretization: Q1 finite elements on a uniform n x n x n grid,
%   h = 1/(n+1).
%
%       K   = kron(kron(K1D,M1D),M1D) + kron(kron(M1D,K1D),M1D)
%             + kron(kron(M1D,M1D),K1D)   Q1 stiffness (N x N, SPD)
%       M_c = kron(kron(M1D,M1D),M1D)     consistent Q1 mass (N x N, SPD)
%       N   = n^3 interior DOFs
%
%   This file uses the consistent Q1 mass matrix M_c = M1D (x) M1D (x) M1D,
%   matching the discretization in the paper.
%
%   Runs a mesh sweep over n in n_vals, comparing solvers:
%
%       (a) MOSEK (interior-point, reference)
%           (skipped for N > mosek_N_max; Schur complement fill O(N^{5/3})
%           exhausts RAM for large N)
%       (b) Gurobi (interior-point + active-set, commercial)
%       (c) HiGHS (interior-point, open-source)
%       (d) DAQP (active-set, dense; skipped for N > daqp_N_max)
%       (e) OSQP (ADMM, first-order)
%       (f) PIQP (proximal interior point, sparse)
%       (g) di_qp_eq with the default internal direct Cholesky of K^2+M_c^2
%           (skipped for N > chol_N_max; impractical in 3D for large N)
%       (h) di_qp on the null-space-reduced u-only QP; equality constraint
%           K*y = M_c*u eliminated by substituting y = K^{-1}*M_c*u.
%           K is prefactored once per mesh via sparse Cholesky with AMD
%           reordering; Q_red is applied as a function handle.
%       (i) rapdhg_qp on the same null-space-reduced u-only QP (same K
%           factorization, same Q_red function handle).  First-order method;
%           uses relative KKT tolerance 1e-6.
%
%   NOTE: These MATLAB files were built with the assistance of Claude
%   (Anthropic) and may contain minor deviations from the paper.
%
%   Solver (c) eliminates the equality constraint K*y = M_c*u by the
%   substitution y = K^{-1}*M_c*u, yielding the reduced QP
%
%       min_u  (1/2)*u'*Q_red*u + c_red'*u   s.t.  u_a <= u <= u_b,
%
%   where Q_red*u = M_c*(K^{-1}*(M_c*(K^{-1}*(M_c*u)))) + beta*M_c*u and
%   c_red = -M_c*(K^{-1}*(M_c*yd)).  Primal feasibility of the equality
%   constraint is satisfied by construction.  Each application of Q_red
%   costs three sparse mat-vecs and two triangular solves with L_K.
%
% -------------------------------------------------------------------------
%   FUNCTIONS CALLED
%       mosekopt         (MOSEK Optimization Toolbox, interior-point)
%       di_qp_eq         (equality + inequality form)
%       di_qp            (inequality form, null-space reduction)
%
%   LOCAL FUNCTIONS
%       build_q1_fem_3d  assemble 3-D Q1 stiffness and consistent mass matrices
%       compute_kkt      compute KKT residuals for one solver
%       k_chol_solve     apply K^{-1} via stored sparse Cholesky factor
%       print_row        print one solver row in the summary table
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

n_vals       = [4, 8, 16, 64];  % interior grid points per dimension
mosek_N_max  = 1^2;         % skip MOSEK for N > this (Schur complement O(N^{5/3}) fill)
chol_N_max   = 1^2;         % skip direct Chol for N > this (impractical in 3D)
gurobi_N_max   = 1^3;       % skip Gurobi for N > this
highs_N_max    = inf;   % skip HiGHS for N > this
daqp_N_max     = 1^3;   % skip DAQP for N > this (dense active-set; only practical for small N)
osqp_N_max     = 1^3;   % skip OSQP for N > this
piqp_N_max     = 1^3;  % skip PIQP for N > this
rapdhg_N_max   = 1^3;  % skip rAPDHG for N > this

% control regularization and bounds
beta = 1e-2;
u_a  = 0;
u_b = 1;

tol_kkt = 1e-08;

fprintf('Benchmark: 3-D Poisson OC, control bounds only, consistent Q1 mass, mesh sweep.\n')
fprintf('  beta = %g,  u in [%g, %g]\n', beta, u_a, u_b)
fprintf('  Desired state: Gaussian bump at (0.5, 0.5, 0.5).\n\n')


%% Solver availability

have_mosek  = exist('mosekopt',  'file')  > 0;
have_gurobi = exist('gurobi',    'file')  > 0;
have_highs  = exist('callhighs', 'file')  > 0;
have_daqp   = exist('daqp',      'class') > 0;
have_osqp   = exist('osqp',      'class') > 0;
have_piqp   = exist('piqp',      'file')  > 0;
have_rapdhg = exist('rapdhg_qp', 'file')  > 0;

avail_rows = { ...
    'MOSEK',   have_mosek,  mosek_N_max;  ...
    'Gurobi',  have_gurobi, gurobi_N_max; ...
    'HiGHS',   have_highs,  highs_N_max;  ...
    'DAQP',    have_daqp,   daqp_N_max;   ...
    'OSQP',    have_osqp,   osqp_N_max;   ...
    'PIQP',    have_piqp,   piqp_N_max;   ...
    'rAPDHG',  have_rapdhg, rapdhg_N_max; ...
    'Chol',    true,        chol_N_max;   ...
    'NSred',   true,        Inf           };

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
res = struct( ...
    'n',          zeros(nn, 1), ...
    'N',          zeros(nn, 1), ...
    't_mosek',    nan(nn, 1),   ...
    't_gurobi',   zeros(nn, 1), ...
    't_chol',     nan(nn, 1),   ...
    't_cg',       zeros(nn, 1), ...
    'fval_mosek', nan(nn, 1),   ...
    'fval_gurobi',zeros(nn, 1), ...
    'fval_chol',  nan(nn, 1),   ...
    'fval_cg',    zeros(nn, 1), ...
    'kkt_stat_mosek',   nan(nn, 1),   ...
    'kkt_pfeas_mosek',  nan(nn, 1),   ...
    'kkt_dfeas_mosek',  nan(nn, 1),   ...
    'fgap_mosek',       nan(nn, 1),   ...
    'kkt_stat_gurobi',  zeros(nn, 1), ...
    'kkt_pfeas_gurobi', zeros(nn, 1), ...
    'kkt_dfeas_gurobi', zeros(nn, 1), ...
    'fgap_gurobi',      zeros(nn, 1), ...
    't_highs',          zeros(nn, 1), ...
    'fval_highs',       zeros(nn, 1), ...
    'kkt_stat_highs',   zeros(nn, 1), ...
    'kkt_pfeas_highs',  zeros(nn, 1), ...
    'kkt_dfeas_highs',  zeros(nn, 1), ...
    'fgap_highs',       zeros(nn, 1), ...
    't_daqp',           zeros(nn, 1), ...
    'fval_daqp',        zeros(nn, 1), ...
    'kkt_stat_daqp',    zeros(nn, 1), ...
    'kkt_pfeas_daqp',   zeros(nn, 1), ...
    'kkt_dfeas_daqp',   zeros(nn, 1), ...
    'fgap_daqp',        zeros(nn, 1), ...
    't_osqp',           zeros(nn, 1), ...
    'fval_osqp',        zeros(nn, 1), ...
    'kkt_stat_osqp',    zeros(nn, 1), ...
    'kkt_pfeas_osqp',   zeros(nn, 1), ...
    'kkt_dfeas_osqp',   zeros(nn, 1), ...
    'fgap_osqp',        zeros(nn, 1), ...
    't_piqp',           zeros(nn, 1), ...
    'fval_piqp',        zeros(nn, 1), ...
    'kkt_stat_piqp',    zeros(nn, 1), ...
    'kkt_pfeas_piqp',   zeros(nn, 1), ...
    'kkt_dfeas_piqp',   zeros(nn, 1), ...
    'fgap_piqp',        zeros(nn, 1), ...
    'kkt_stat_chol',    nan(nn, 1),   ...
    'kkt_pfeas_chol',   nan(nn, 1),   ...
    'kkt_dfeas_chol',   nan(nn, 1),   ...
    'fgap_chol',        nan(nn, 1),   ...
    'kkt_stat_cg',      zeros(nn, 1), ...
    'kkt_pfeas_cg',     zeros(nn, 1), ...
    'kkt_dfeas_cg',     zeros(nn, 1), ...
    'fgap_cg',          zeros(nn, 1), ...
    'cg_calls',   zeros(nn, 1), ...
    'cg_iters',   zeros(nn, 1), ...
    't_rapdhg',          nan(nn, 1), ...
    'fval_rapdhg',       nan(nn, 1), ...
    'kkt_stat_rapdhg',   nan(nn, 1), ...
    'kkt_pfeas_rapdhg',  nan(nn, 1), ...
    'kkt_dfeas_rapdhg',  nan(nn, 1), ...
    'fgap_rapdhg',       nan(nn, 1), ...
    'flag_rapdhg',       nan(nn, 1) );


%% Loop over mesh sizes

for idx = 1:nn

    n = n_vals(idx);
    N = n^3;
    h = 1/(n+1);

    run_mosek   = have_mosek  && (N <= mosek_N_max);
    run_gurobi  = have_gurobi && (N <= gurobi_N_max);
    run_highs   = have_highs  && (N <= highs_N_max);
    run_daqp    = have_daqp   && (N <= daqp_N_max);
    run_osqp    = have_osqp   && (N <= osqp_N_max);
    run_piqp    = have_piqp   && (N <= piqp_N_max);
    run_chol    = (N <= chol_N_max);
    run_rapdhg  = have_rapdhg && (N <= rapdhg_N_max);

    fprintf('=== n = %d  (N = %d DOFs,  h = 1/%d) ===\n\n', n, N, n+1)


    %% Grid and desired state

    x1vec = (1:n)' * h;
    [X1, X2, X3] = ndgrid(x1vec, x1vec, x1vec);
    Yhat = exp(-64 * ((X1 - 0.5).^2 + (X2 - 0.5).^2 + (X3 - 0.5).^2));
    yd   = Yhat(:);


    %% FEM matrices: Q1 stiffness and consistent Q1 mass

    [K, M_mat] = build_q1_fem_3d(n);


    %% QP data

    H    = blkdiag(M_mat, beta * M_mat);
    fvec = [-M_mat * yd; sparse(N, 1)];
    Aeq  = [K, -M_mat];
    beq  = zeros(N, 1);
    lb   = [-inf(N, 1);  u_a * ones(N, 1)];
    ub   = [ inf(N, 1);  u_b * ones(N, 1)];


    %% Constraint matrices for di_qp_eq (control bounds only)

    E_eq = [K, -M_mat];                          % N x 2N

    % Row blocks (each N rows): u <= u_b, u >= u_a.
    A_di_eq = [sparse(N,N),  speye(N);
               sparse(N,N), -speye(N)];           % 2N x 2N
    b_di_eq = [u_b * ones(N,1); -u_a * ones(N,1)];

    tol_di_eq = 1e-08; 


    %% Feasible starting point: u0 = midpoint, y0 = K\(M_c*u0)

    u0 = 0.5*(u_a + u_b) * ones(N, 1);
    y0 = K \ (M_mat * u0);
    x0 = [y0; u0];


    % KKT diagnostics reported for each solver:
    %
    %   kkt_stat  = ||mu - beta*u||_inf restricted to components of u
    %               strictly interior to (u_a, u_b).  Measures first-order
    %               stationarity of the control; zero at an exact KKT point.
    %   kkt_pfeas = max(||K*y - M_c*u||_inf,  max(0, max([u-u_b; u_a-u]))).
    %               Primal feasibility: max of the PDE equality residual and
    %               the bound-constraint violation.
    %   kkt_dfeas = max of: the state stationarity ||M*(y-yd) + K*mu||_inf
    %               (zero by construction for di_qp_eq and NSred; computed
    %               for MOSEK), max(0, -r_u) over components at u_b, and
    %               max(0, r_u) over components at u_a.  At an exact KKT
    %               point, r_u >= 0 at u_b and r_u <= 0 at u_a.


    %% 1. MOSEK

    if run_mosek
        tic
        [qi, qj, qv] = find(tril(H));
        prob_mosek.qosubi   = qi;
        prob_mosek.qosubj   = qj;
        prob_mosek.qoval    = qv;
        prob_mosek.c        = full(fvec);
        prob_mosek.a        = Aeq;
        prob_mosek.blc      = beq;
        prob_mosek.buc      = beq;
        prob_mosek.blx      = lb;
        prob_mosek.bux      = ub;
        prob_mosek.iparam.INTPNT_ORDER_METHOD = 'MSK_ORDER_METHOD_FORCEGRAPHPAR';
        param.MSK_IPAR_LOG = 0;

        fprintf('  Running MOSEK...\n')
        [~, res_mosek] = mosekopt('minimize echo(0)', prob_mosek, param);
        t_mosek = toc;

        x_mosek      = res_mosek.sol.itr.xx;
        fval_mosek   = res_mosek.sol.itr.pobjval;
        flag_mosek   = res_mosek.rcode;
        lambda_mosek = -res_mosek.sol.itr.y;   % MOSEK uses opposite sign convention

        y_mosek = x_mosek(1:N);
        u_mosek = x_mosek(N+1:2*N);

        fgap_mosek = res_mosek.sol.itr.pobjval - res_mosek.sol.itr.dobjval;
        [kkt_stat_mosek, kkt_pfeas_mosek, kkt_dfeas_mosek] = ...
            compute_kkt(y_mosek, u_mosek, lambda_mosek, K, M_mat, yd, beta, u_a, u_b, tol_kkt, true);

        fprintf('  MOSEK:  t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (flag=%d)\n\n', ...
                t_mosek, fval_mosek, kkt_stat_mosek, kkt_pfeas_mosek, kkt_dfeas_mosek, fgap_mosek, flag_mosek)

        res.t_mosek(idx)         = t_mosek;
        res.fval_mosek(idx)      = fval_mosek;
        res.kkt_stat_mosek(idx)  = kkt_stat_mosek;
        res.kkt_pfeas_mosek(idx) = kkt_pfeas_mosek;
        res.kkt_dfeas_mosek(idx) = kkt_dfeas_mosek;
        res.fgap_mosek(idx)      = fgap_mosek;
    else
        if ~have_mosek
            fprintf('  Skipping MOSEK (not found).\n\n')
        else
            fprintf('  Skipping MOSEK (N = %d > %d).\n\n', N, mosek_N_max)
        end
    end


    %% 2. Gurobi (interior-point)
    %
    %   Full (y,u) QP: same formulation as MOSEK.
    %   Gurobi minimizes x'*Q*x + obj'*x  (no 1/2 factor), so pass Q = H/2.
    %   result.pi for an equality constraint gives the Lagrange multiplier mu
    %   directly (no sign flip; satisfies H*x + f + Aeq'*mu = 0).

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
        % Duality gap: mu'*(beq - Aeq*x) + nu_lb'*(x - lb_u) + nu_ub'*(ub_u - x).
        % rc = Hx+f+Aeq'*mu = nu_lb - nu_ub at KKT; only u-block has finite bounds.
        rc_u_g = result_g.rc(N+1:2*N);
        fgap_gurobi = result_g.pi' * (beq - Aeq * result_g.x) ...
                    + max(0,  rc_u_g)' * (result_g.x(N+1:2*N) - u_a) ...
                    + max(0, -rc_u_g)' * (u_b - result_g.x(N+1:2*N));
        t_gurobi = toc;

        if ~strcmp(result_g.status, 'OPTIMAL')
            error('Gurobi did not find an optimal solution (status: %s).', result_g.status)
        end

        x_gurobi      = result_g.x;
        fval_gurobi   = result_g.objval;
        lambda_gurobi = result_g.pi;

        y_gurobi = x_gurobi(1:N);
        u_gurobi = x_gurobi(N+1:2*N);

        [kkt_stat_gurobi, kkt_pfeas_gurobi, kkt_dfeas_gurobi] = ...
            compute_kkt(y_gurobi, u_gurobi, lambda_gurobi, K, M_mat, yd, beta, u_a, u_b, tol_kkt, true);

        fprintf('  Gurobi: t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (status=%s)\n\n', ...
                t_gurobi, fval_gurobi, kkt_stat_gurobi, kkt_pfeas_gurobi, kkt_dfeas_gurobi, fgap_gurobi, result_g.status)

        res.t_gurobi(idx)          = t_gurobi;
        res.fval_gurobi(idx)       = fval_gurobi;
        res.kkt_stat_gurobi(idx)   = kkt_stat_gurobi;
        res.kkt_pfeas_gurobi(idx)  = kkt_pfeas_gurobi;
        res.kkt_dfeas_gurobi(idx)  = kkt_dfeas_gurobi;
        res.fgap_gurobi(idx)       = fgap_gurobi;
    else
        if ~have_gurobi
            fprintf('  Skipping Gurobi (not found).\n\n')
        else
            fprintf('  Skipping Gurobi (N = %d > %g).\n\n', N, gurobi_N_max)
        end
    end


    %% 3. HiGHS (interior-point / active-set)
    %
    %   Full (y,u) QP: same formulation as MOSEK and Gurobi.
    %   HiGHS uses (1/2)x'Qx + c'x convention; pass H directly (no scaling).
    %   soln.row_dual for equality L=U=beq: KKT adjoint mu = -soln.row_dual
    %   (sign flip required; same convention as Gurobi's pi for inequalities).

    if run_highs
        tic
        opts_h.output_flag                  = false;
        opts_h.solver                       = "ipm";   % active-set cycles on this problem
        opts_h.ipm_optimality_tolerance     = 1e-10;
        opts_h.primal_feasibility_tolerance = 1e-10;
        opts_h.dual_feasibility_tolerance   = 1e-10;
        fprintf('  Running HiGHS (IPM)...\n')
        
        [soln_h, info_h] = callhighs(full(fvec), Aeq, beq, beq, lb, ub, H, [], opts_h);
        % Duality gap: mu'*(beq - Aeq*x) + nu_lb'*(x - lb_u) + nu_ub'*(ub_u - x).
        % col_dual = nu_lb - nu_ub at KKT; only u-block has finite bounds.
        rc_u_h     = soln_h.col_dual(N+1:2*N);
        mu_h       = -soln_h.row_dual;
        fgap_highs = mu_h' * (beq - Aeq * soln_h.col_value) ...
                   + max(0,  rc_u_h)' * (soln_h.col_value(N+1:2*N) - u_a) ...
                   + max(0, -rc_u_h)' * (u_b - soln_h.col_value(N+1:2*N));
        t_highs = toc;

        if ~strcmp(info_h.model_status_string, 'Optimal')
            error('HiGHS did not find an optimal solution (status: %s).', info_h.model_status_string)
        end

        x_highs      = soln_h.col_value;
        fval_highs   = info_h.objective_function_value;
        lambda_highs = -soln_h.row_dual;

        y_highs = x_highs(1:N);
        u_highs = x_highs(N+1:2*N);

        [kkt_stat_highs, kkt_pfeas_highs, kkt_dfeas_highs] = ...
            compute_kkt(y_highs, u_highs, lambda_highs, K, M_mat, yd, beta, u_a, u_b, tol_kkt, true);

        fprintf('  HiGHS:  t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (status=%s)\n\n', ...
                t_highs, fval_highs, kkt_stat_highs, kkt_pfeas_highs, kkt_dfeas_highs, fgap_highs, info_h.model_status_string)

        res.t_highs(idx)         = t_highs;
        res.fval_highs(idx)      = fval_highs;
        res.kkt_stat_highs(idx)  = kkt_stat_highs;
        res.kkt_pfeas_highs(idx) = kkt_pfeas_highs;
        res.kkt_dfeas_highs(idx) = kkt_dfeas_highs;
        res.fgap_highs(idx)      = fgap_highs;
    else
        if ~have_highs
            fprintf('  Skipping HiGHS (not found).\n\n')
        else
            fprintf('  Skipping HiGHS (N = %d > %g).\n\n', N, highs_N_max)
        end
    end


    %% 4. DAQP (active-set, dense)
    %
    %   Full (y,u) QP: same formulation as MOSEK, Gurobi, and HiGHS.
    %   DAQP uses (1/2)x'Hx + f'x convention; pass H directly (no scaling).
    %   Simple bounds (ms = 2N): y in (-Inf,Inf), u in [u_a, u_b].
    %   Equality constraint K*y = M_c*u is encoded as a general constraint
    %   with blower = bupper = beq (double-sided bound at zero, sense = 0).
    %   Dual sign convention: DAQP uses the STANDARD KKT convention (same as
    %   OSQP) — no sign flip.  info.lambda(1:2N) are bound duals and
    %   info.lambda(2N+1:3N) is the equality adjoint mu directly.

    if run_daqp
        tic
        ms_d     = 2*N;
        bupper_d = [inf(N,1);    u_b*ones(N,1); full(beq)];
        blower_d = [-inf(N,1);   u_a*ones(N,1); full(beq)];
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
        lambda_u_d = lambda_d(N+1:2*N);          % control bound multipliers (standard KKT sign)

        fgap_daqp = mu_eq_d' * (beq - Aeq * x_d) ...
                  + max(0, -lambda_u_d)' * (x_d(N+1:2*N) - u_a) ...
                  + max(0,  lambda_u_d)' * (u_b - x_d(N+1:2*N));

        y_d = x_d(1:N);
        u_d = x_d(N+1:2*N);

        [kkt_stat_d, kkt_pfeas_d, kkt_dfeas_d] = ...
            compute_kkt(y_d, u_d, mu_eq_d, K, M_mat, yd, beta, u_a, u_b, tol_kkt, true);

        fprintf('  DAQP:   t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (exitflag=%d)\n\n', ...
                t_daqp, fval_d, kkt_stat_d, kkt_pfeas_d, kkt_dfeas_d, fgap_daqp, exitflag_d)

        res.t_daqp(idx)         = t_daqp;
        res.fval_daqp(idx)      = fval_d;
        res.kkt_stat_daqp(idx)  = kkt_stat_d;
        res.kkt_pfeas_daqp(idx) = kkt_pfeas_d;
        res.kkt_dfeas_daqp(idx) = kkt_dfeas_d;
        res.fgap_daqp(idx)      = fgap_daqp;
    else
        if ~have_daqp
            fprintf('  Skipping DAQP (not found).\n\n')
        else
            fprintf('  Skipping DAQP (N = %d > %g).\n\n', N, daqp_N_max)
        end
    end


    %% 5. OSQP (ADMM, first-order)
    %
    %   Full (y,u) QP: same formulation as MOSEK, Gurobi, and HiGHS.
    %   OSQP uses (1/2)x'Px + q'x convention; pass P = triu(H).
    %   Variable bounds on u are encoded as rows of the constraint matrix
    %   (OSQP has no separate variable-bound argument):
    %       A_osqp = [Aeq; [0, I]]  (N equality + N control-bound rows)
    %       l_osqp = [beq; u_a*ones]
    %       ub_osqp = [beq; u_b*ones]
    %   Dual sign convention: OSQP uses the STANDARD KKT convention —
    %   no sign flip.  res_osqp.y(1:N) is the equality adjoint directly,
    %   and res_osqp.y(N+1:2*N) >= 0 at active upper control bound (u = u_b),
    %   <= 0 at active lower (u = u_a).  This is OPPOSITE to HiGHS and
    %   HiGHS (row_dual), which requires a sign flip.

    if run_osqp
        tic
        A_osqp  = [Aeq; sparse(N,N), speye(N)];
        l_osqp  = [beq;    u_a * ones(N,1)];
        ub_osqp = [beq;    u_b * ones(N,1)];

        prob_osqp = osqp;
        prob_osqp.setup(triu(H), full(fvec), A_osqp, l_osqp, ub_osqp, ...
            'verbose',  false, ...
            'eps_abs',  1e-06, ...
            'eps_rel',  1e-06, ...
            'polish',   true,  ...
            'max_iter', 10000);

        fprintf('  Running OSQP (ADMM)...\n')
        
        res_osqp = prob_osqp.solve();
        t_osqp = toc;

        if res_osqp.info.status_val ~= 1
            warning('OSQP did not find an optimal solution (status: %s).', res_osqp.info.status)
        end

        x_osqp    = res_osqp.x;
        fval_osqp = res_osqp.info.obj_val;

        % Dual variables (no sign flip — OSQP uses standard KKT convention).
        mu_eq_osqp = res_osqp.y(1:N);        % equality adjoint
        y_u_osqp   = res_osqp.y(N+1:2*N);   % control bound dual

        % Duality gap (sign convention: y_u >= 0 at u_b, <= 0 at u_a).
        fgap_osqp = mu_eq_osqp' * (beq - Aeq * x_osqp) ...
                  + max(0, -y_u_osqp)' * (x_osqp(N+1:2*N) - u_a) ...
                  + max(0,  y_u_osqp)' * (u_b - x_osqp(N+1:2*N));

        y_osqp_sol = x_osqp(1:N);
        u_osqp_sol = x_osqp(N+1:2*N);

        [kkt_stat_osqp, kkt_pfeas_osqp, kkt_dfeas_osqp] = ...
            compute_kkt(y_osqp_sol, u_osqp_sol, mu_eq_osqp, K, M_mat, yd, beta, u_a, u_b, tol_kkt, true);

        fprintf('  OSQP:   t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (status=%s)\n\n', ...
                t_osqp, fval_osqp, kkt_stat_osqp, kkt_pfeas_osqp, kkt_dfeas_osqp, fgap_osqp, res_osqp.info.status)

        res.t_osqp(idx)          = t_osqp;
        res.fval_osqp(idx)       = fval_osqp;
        res.kkt_stat_osqp(idx)   = kkt_stat_osqp;
        res.kkt_pfeas_osqp(idx)  = kkt_pfeas_osqp;
        res.kkt_dfeas_osqp(idx)  = kkt_dfeas_osqp;
        res.fgap_osqp(idx)       = fgap_osqp;
    else
        if ~have_osqp
            fprintf('  Skipping OSQP (not found).\n\n')
        else
            fprintf('  Skipping OSQP (N = %d > %g).\n\n', N, osqp_N_max)
        end
    end


    %% 6. PIQP (proximal interior point)
    %
    %   Full (y,u) QP: same formulation as MOSEK, Gurobi, and HiGHS.
    %   PIQP uses (1/2)x'Px + c'x convention; pass H directly (no scaling).
    %   Variable bounds lb <= x <= ub are passed as x_l, x_u (no rows in G).
    %   Equality constraint K*y = M_c*u is passed as A*x = b directly.
    %   There are no general inequality constraints (G is empty).
    %
    %   Dual sign convention: PIQP uses the STANDARD KKT convention.
    %   result.y is the equality adjoint mu directly (no sign flip).
    %   result.z_bl and result.z_bu are the box-bound duals (both >= 0):
    %       z_bu(N+1:2N) >= 0 at active upper control bound (u = u_b).
    %       z_bl(N+1:2N) >= 0 at active lower control bound (u = u_a).
    %   Stationarity: H*x + fvec + Aeq'*y + (z_bu - z_bl) = 0.
    %
    %   QUIRK — status is a string, not an integer.  Use strcmp to check:
    %       strcmp(result.info.status, 'solved')  for optimality.

    if run_piqp
        tic
        G_p   = sparse(0, 2*N);   % no general inequality constraints
        h_l_p = zeros(0, 1);
        h_u_p = zeros(0, 1);

        solver_p = piqp('sparse');
        solver_p.update_settings( ...
            'verbose',         false, ...
            'eps_abs',         1e-9,  ...
            'eps_rel',         1e-9,  ...
            'compute_timings', true);
        solver_p.setup(H, full(fvec), Aeq, full(beq), G_p, h_l_p, h_u_p, full(lb), full(ub));

        fprintf('  Running PIQP (proximal IPM)...\n')
        
        result_p = solver_p.solve();
        t_piqp = toc;

        if ~strcmp(result_p.info.status, 'solved')
            warning('PIQP did not find an optimal solution (status: %s).', result_p.info.status)
        end

        x_piqp      = result_p.x;
        fval_piqp   = result_p.info.primal_obj;
        mu_eq_piqp  = result_p.y;            % equality adjoint (standard KKT sign; no flip)
        z_bl_u      = result_p.z_bl(N+1:2*N);
        z_bu_u      = result_p.z_bu(N+1:2*N);

        fgap_piqp = mu_eq_piqp' * (beq - Aeq * x_piqp) ...
                  + z_bl_u' * (x_piqp(N+1:2*N) - u_a) ...
                  + z_bu_u' * (u_b - x_piqp(N+1:2*N));

        y_piqp = x_piqp(1:N);
        u_piqp = x_piqp(N+1:2*N);

        [kkt_stat_piqp, kkt_pfeas_piqp, kkt_dfeas_piqp] = ...
            compute_kkt(y_piqp, u_piqp, mu_eq_piqp, K, M_mat, yd, beta, u_a, u_b, tol_kkt, true);

        fprintf('  PIQP:   t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (status=%s)\n\n', ...
                t_piqp, fval_piqp, kkt_stat_piqp, kkt_pfeas_piqp, kkt_dfeas_piqp, fgap_piqp, result_p.info.status)

        res.t_piqp(idx)         = t_piqp;
        res.fval_piqp(idx)      = fval_piqp;
        res.kkt_stat_piqp(idx)  = kkt_stat_piqp;
        res.kkt_pfeas_piqp(idx) = kkt_pfeas_piqp;
        res.kkt_dfeas_piqp(idx) = kkt_dfeas_piqp;
        res.fgap_piqp(idx)      = fgap_piqp;
    else
        if ~have_piqp
            fprintf('  Skipping PIQP (not found).\n\n')
        else
            fprintf('  Skipping PIQP (N = %d > %g).\n\n', N, piqp_N_max)
        end
    end


    %% 7. di_qp_eq, direct Cholesky of K^2+M_c^2

    if run_chol
        fprintf('  Running di_qp_eq (direct Cholesky)...\n')
        tic
        [x_chol, ~, mu_chol, fval_chol, flag_chol, fgap_chol] = ...
            di_qp_eq(A_di_eq, b_di_eq, fvec, H, E_eq, beq, x0, tol_di_eq, [], [], [], @apgd_lsqnonneg);
        t_chol = toc;

        y_chol = x_chol(1:N);
        u_chol = x_chol(N+1:2*N);

        [kkt_stat_chol, kkt_pfeas_chol, kkt_dfeas_chol] = ...
            compute_kkt(y_chol, u_chol, mu_chol, K, M_mat, yd, beta, u_a, u_b, tol_kkt, false);

        fprintf('  Chol:   t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (flag=%d)\n\n', ...
                t_chol, fval_chol, kkt_stat_chol, kkt_pfeas_chol, kkt_dfeas_chol, fgap_chol, flag_chol)

        res.t_chol(idx)         = t_chol;
        res.fval_chol(idx)      = fval_chol;
        res.kkt_stat_chol(idx)  = kkt_stat_chol;
        res.kkt_pfeas_chol(idx) = kkt_pfeas_chol;
        res.kkt_dfeas_chol(idx) = kkt_dfeas_chol;
        res.fgap_chol(idx)      = fgap_chol;
    else
        fprintf('  Skipping direct Cholesky (N = %d > %d).\n\n', N, chol_N_max)
    end


    %% 8. Prefactor K, then solve null-space-reduced u-only QP via di_qp

    tic
    % Factor K once: K(perm_K, perm_K) = L_K * L_K'  (AMD reordering).
    perm_K = dissect(K);
    [L_K, chol_flag] = chol(K(perm_K,perm_K), 'lower');
    if chol_flag ~= 0, error('chol(K) failed under dissect ordering.'); end
    t_factor_K = toc;

    % Null-space elimination: y = K^{-1}*M_c*u satisfies K*y = M_c*u.
    % Reduced QP in u: min (1/2)*u'*Q_red*u + c_red'*u  s.t.  u_a <= u <= u_b.
    Ksolve = @(v) k_chol_solve(L_K, perm_K, v);
    Q_red  = @(u) M_mat*(Ksolve(M_mat*(Ksolve(M_mat*u)))) + beta*(M_mat*u);
    c_red  = -M_mat*(Ksolve(M_mat*yd));
    A_red  = [speye(N); -speye(N)];
    b_red  = [u_b*ones(N,1); -u_a*ones(N,1)];

    fprintf('  Running di_qp (null-space reduction)...\n')

    tic
    [u_cg, p_cg, fval_red_cg, flag_cg] = di_qp(A_red, b_red, c_red, ...
        Q_red, u0, tol_di_eq, [], @apgd_lsqnonneg);
    t_cg = toc + t_factor_K;

    % Recover state; equality constraint satisfied by construction.
    y_cg = Ksolve(M_mat * u_cg);
    x_cg = [y_cg; u_cg];

    fval_cg = fval_red_cg;

    % Primal-dual gap for the reduced inequality constraints.
    fgap_cg = p_cg' * (b_red - A_red*u_cg);

    % Equality dual from adjoint equation M_c*(y - yd) + K*mu = 0.
    mu_cg = Ksolve(M_mat*(yd - y_cg));

    ub_viol_cg = max(0, max(u_cg - u_b));
    lb_viol_cg = max(0, max(u_a - u_cg));
    fprintf('  NSred diagnostics: min(u)=%.4f, max(u)=%.4f, ub_viol=%.2e, lb_viol=%.2e, n_ub_viol=%d, n_lb_viol=%d\n', ...
            min(u_cg), max(u_cg), ub_viol_cg, lb_viol_cg, ...
            sum(u_cg > u_b), sum(u_cg < u_a));

    [kkt_stat_cg, kkt_pfeas_cg, kkt_dfeas_cg] = ...
        compute_kkt(y_cg, u_cg, mu_cg, K, M_mat, yd, beta, u_a, u_b, tol_kkt, false);

    fprintf('  NSred:  t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (flag=%d)\n\n', ...
            t_cg, fval_cg, kkt_stat_cg, kkt_pfeas_cg, kkt_dfeas_cg, fgap_cg, flag_cg)


    %% 9. rAPDHG on the null-space-reduced u-only QP
    %
    %   Reuses the same K factorization, Q_red function handle, c_red,
    %   A_red, b_red, and warm start u0 as the NSred di_qp run above.
    %   Relative KKT tolerance 1e-6 (first-order method; not comparable to
    %   the absolute 1e-8 used by di_qp and the IPM solvers).

    if run_rapdhg
        fprintf('  Running rAPDHG (null-space reduction)...\n')
        tic
        [u_rapdhg, p_rapdhg, fval_red_rapdhg, flag_rapdhg] = ...
            rapdhg_qp(A_red, b_red, c_red, Q_red, u0, [], 1e-6);
        t_rapdhg = toc + t_factor_K;

        y_rapdhg   = Ksolve(M_mat * u_rapdhg);
        fval_rapdhg = fval_red_rapdhg;
        fgap_rapdhg = p_rapdhg' * (b_red - A_red * u_rapdhg);
        mu_rapdhg   = Ksolve(M_mat * (yd - y_rapdhg));

        [kkt_stat_rapdhg, kkt_pfeas_rapdhg, kkt_dfeas_rapdhg] = ...
            compute_kkt(y_rapdhg, u_rapdhg, mu_rapdhg, K, M_mat, yd, beta, u_a, u_b, tol_kkt, false);

        fprintf('  rAPDHG: t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (flag=%d)\n\n', ...
                t_rapdhg, fval_rapdhg, kkt_stat_rapdhg, kkt_pfeas_rapdhg, kkt_dfeas_rapdhg, fgap_rapdhg, flag_rapdhg)

        res.t_rapdhg(idx)         = t_rapdhg;
        res.fval_rapdhg(idx)      = fval_rapdhg;
        res.kkt_stat_rapdhg(idx)  = kkt_stat_rapdhg;
        res.kkt_pfeas_rapdhg(idx) = kkt_pfeas_rapdhg;
        res.kkt_dfeas_rapdhg(idx) = kkt_dfeas_rapdhg;
        res.fgap_rapdhg(idx)      = fgap_rapdhg;
        res.flag_rapdhg(idx)      = flag_rapdhg;
    else
        if ~have_rapdhg
            fprintf('  Skipping rAPDHG (not found).\n\n')
        else
            fprintf('  Skipping rAPDHG (N = %d > %g).\n\n', N, rapdhg_N_max)
        end
    end


    %% Store results

    res.n(idx)            = n;
    res.N(idx)            = N;
    res.t_cg(idx)         = t_cg;
    res.fval_cg(idx)      = fval_cg;
    res.kkt_stat_cg(idx)  = kkt_stat_cg;
    res.kkt_pfeas_cg(idx) = kkt_pfeas_cg;
    res.kkt_dfeas_cg(idx) = kkt_dfeas_cg;
    res.fgap_cg(idx)      = fgap_cg;

end  % mesh loop


%% Build data matrices for per-mesh SGM rankings
% sgm_zero_init: true for solvers whose res fields are initialised to 0
%   (not NaN) when skipped, so t == 0 is the skip sentinel.

sgm_solver_names = {'NSred (di_qp)','Chol (di_qp_eq)','MOSEK','Gurobi','HiGHS', ...
                    'OSQP','PIQP','NSred (rAPDHG)','DAQP'};
sgm_field_pfx    = {'cg','chol','mosek','gurobi','highs','osqp','piqp','rapdhg','daqp'};
sgm_zero_init    = logical([0,0,0,1,1,1,1,0,1]);
ns = numel(sgm_solver_names);

T_sgm = nan(ns, nn);
P_sgm = nan(ns, nn);
D_sgm = nan(ns, nn);
S_sgm = nan(ns, nn);
G_sgm = nan(ns, nn);
for j = 1 : ns
    pf  = sgm_field_pfx{j};
    tv  = res.(['t_' pf])';                   % 1 × nn
    if sgm_zero_init(j)
        ok = tv ~= 0;
    else
        ok = ~isnan(tv);
    end
    T_sgm(j, ok) = tv(ok);
    P_sgm(j, ok) = res.(['kkt_pfeas_' pf])(ok)';
    D_sgm(j, ok) = res.(['kkt_dfeas_' pf])(ok)';
    S_sgm(j, ok) = res.(['kkt_stat_'  pf])(ok)';
    G_sgm(j, ok) = abs(res.(['fgap_'  pf])(ok)');
end


%% Summary table

hdr = sprintf('  %-20s  %11s  %12s  %12s  %12s  %12s', ...
    'Solver', 'Runtime (s)', 'Primal res', 'Dual res', 'Stat. compl.', 'Duality gap');
sep = ['  ' repmat('-', 1, length(hdr) - 2)];

fprintf('\nSummary\n')

for idx = 1:nn
    fprintf('\n=== n = %d,  N = %d,  h = 1/%d ===\n\n', ...
            res.n(idx), res.N(idx), res.n(idx) + 1)
    fprintf('%s\n', hdr)
    fprintf('%s\n', sep)

    print_row('NSred (di_qp)', res.t_cg(idx), ...
        [res.kkt_pfeas_cg(idx), res.kkt_dfeas_cg(idx), ...
         res.kkt_stat_cg(idx),  res.fgap_cg(idx)])

    if ~isnan(res.t_chol(idx))
        print_row('Chol (di_qp_eq)', res.t_chol(idx), ...
            [res.kkt_pfeas_chol(idx), res.kkt_dfeas_chol(idx), ...
             res.kkt_stat_chol(idx),  res.fgap_chol(idx)])
    else
        fprintf('  %-20s  %11s  %12s  %12s  %12s  %12s\n', ...
                'Chol (skipped)', '---', '---', '---', '---', '---')
    end

    if ~isnan(res.t_mosek(idx))
        print_row('MOSEK', res.t_mosek(idx), ...
            [res.kkt_pfeas_mosek(idx), res.kkt_dfeas_mosek(idx), ...
             res.kkt_stat_mosek(idx),  res.fgap_mosek(idx)])
    else
        fprintf('  %-20s  %11s  %12s  %12s  %12s  %12s\n', ...
                'MOSEK (skipped)', '---', '---', '---', '---', '---')
    end

    if res.t_gurobi(idx) ~= 0
        print_row('Gurobi', res.t_gurobi(idx), ...
            [res.kkt_pfeas_gurobi(idx), res.kkt_dfeas_gurobi(idx), ...
             res.kkt_stat_gurobi(idx),  res.fgap_gurobi(idx)])
    else
        fprintf('  %-20s  %11s  %12s  %12s  %12s  %12s\n', ...
                'Gurobi (skipped)', '---', '---', '---', '---', '---')
    end

    if res.t_highs(idx) ~= 0
        print_row('HiGHS', res.t_highs(idx), ...
            [res.kkt_pfeas_highs(idx), res.kkt_dfeas_highs(idx), ...
             res.kkt_stat_highs(idx),  res.fgap_highs(idx)])
    else
        fprintf('  %-20s  %11s  %12s  %12s  %12s  %12s\n', ...
                'HiGHS (skipped)', '---', '---', '---', '---', '---')
    end

    if res.t_osqp(idx) ~= 0
        print_row('OSQP', res.t_osqp(idx), ...
            [res.kkt_pfeas_osqp(idx), res.kkt_dfeas_osqp(idx), ...
             res.kkt_stat_osqp(idx),  res.fgap_osqp(idx)])
    else
        fprintf('  %-20s  %11s  %12s  %12s  %12s  %12s\n', ...
                'OSQP (skipped)', '---', '---', '---', '---', '---')
    end

    if res.t_piqp(idx) ~= 0
        print_row('PIQP', res.t_piqp(idx), ...
            [res.kkt_pfeas_piqp(idx), res.kkt_dfeas_piqp(idx), ...
             res.kkt_stat_piqp(idx),  res.fgap_piqp(idx)])
    else
        fprintf('  %-20s  %11s  %12s  %12s  %12s  %12s\n', ...
                'PIQP (skipped)', '---', '---', '---', '---', '---')
    end

    if ~isnan(res.t_rapdhg(idx))
        print_row('NSred (rAPDHG)', res.t_rapdhg(idx), ...
            [res.kkt_pfeas_rapdhg(idx), res.kkt_dfeas_rapdhg(idx), ...
             res.kkt_stat_rapdhg(idx),  res.fgap_rapdhg(idx)])
    else
        fprintf('  %-20s  %11s  %12s  %12s  %12s  %12s\n', ...
                'NSred (rAPDHG skipped)', '---', '---', '---', '---', '---')
    end

    if res.t_daqp(idx) ~= 0
        print_row('DAQP', res.t_daqp(idx), ...
            [res.kkt_pfeas_daqp(idx), res.kkt_dfeas_daqp(idx), ...
             res.kkt_stat_daqp(idx),  res.fgap_daqp(idx)])
    else
        fprintf('  %-20s  %11s  %12s  %12s  %12s  %12s\n', ...
                'DAQP (skipped)', '---', '---', '---', '---', '---')
    end

    % --- Per-mesh SGM ranking (ratios relative to best solver at this n) ---
    % For a single problem instance, SGM_s({r}) = r, so the ratio IS the SGM.
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
    for j = 1 : ns
        fprintf('  %-22s  %s  %s  %s  %s  %s\n', sgm_solver_names{j}, ...
            fmtsgm(rt(j)), fmtsgm(rp(j)), fmtsgm(rd(j)), ...
            fmtsgm(rs(j)), fmtsgm(rg(j)))
    end
    fprintf('\n\n')
end


%% =========================================================================
%  Local functions
%  =========================================================================

function [K, M_mat] = build_q1_fem_3d(n)
%BUILD_Q1_FEM_3D  Assemble 3-D Q1 stiffness and consistent mass matrices.
%   [K, M_mat] = BUILD_Q1_FEM_3D(n) returns the N x N sparse SPD stiffness
%   matrix K and consistent mass matrix M_mat for Q1 finite elements on a
%   uniform n x n x n interior grid, h = 1/(n+1), N = n^3.
    h   = 1 / (n + 1);
    e   = ones(n, 1);
    K1D = spdiags([-e, 2*e, -e], [-1,0,1], n, n) / h;
    M1D = spdiags([ e, 4*e,  e], [-1,0,1], n, n) * (h/6);
    K     = kron(kron(K1D,M1D),M1D) + kron(kron(M1D,K1D),M1D) + kron(kron(M1D,M1D),K1D);
    M_mat = kron(kron(M1D,M1D),M1D);
end

function [kkt_stat, kkt_pfeas, kkt_dfeas] = ...
        compute_kkt(y, u, mu, K, M_mat, yd, beta, u_a, u_b, tol_kkt, full_diag)
%COMPUTE_KKT  Compute KKT residuals for the 3-D Poisson OC problem.
%
%   Inputs
%       y, u        primal state and control
%       mu          equality dual (adjoint / co-state)
%       full_diag   true: include state stationarity ||M(y-yd)+K*mu||_inf
%                   in kkt_dfeas (for solvers that do not enforce it by
%                   construction, e.g. MOSEK).  false: omit (zero by
%                   construction for di_qp_eq and NSred).
%
%   Outputs
%       kkt_stat    ||mu - beta*u||_inf over interior controls
%       kkt_pfeas   max(||K*y - M*u||_inf, bound violation)
%       kkt_dfeas   max of multiplier sign violations (+ state stat. if full_diag)

    r_u      = mu - beta * u;
    interior = (u > u_a + tol_kkt) & (u < u_b - tol_kkt);
    if any(interior)
        kkt_stat = norm(r_u(interior), inf);
    else
        kkt_stat = 0;
    end

    eq_res   = norm(K*y - M_mat*u, inf);
    ineq_res = max(0, max([u - u_b; u_a - u]));
    kkt_pfeas = max(eq_res, ineq_res);

    at_ub = u >= u_b - tol_kkt;
    at_lb = u <= u_a + tol_kkt;
    kkt_dfeas = max(max([0; -r_u(at_ub)]), max([0; r_u(at_lb)]));

    if full_diag
        state_stat = norm(M_mat*(y - yd) + K*mu, inf);
        kkt_dfeas  = max(kkt_dfeas, state_stat);
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
