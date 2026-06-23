%   BENCHMARK_PDE_OC_2D_II     Time-dependent distributed control of the heat
%                              equation on the unit square, control bounds
%                              only, Q1 finite elements in space + backward
%                              Euler in time (space-time QP), mesh sweep in h.
%
%   Reference: Pearson & Gondzio (2017), "Fast interior point solution of
%   quadratic programming problems arising from PDE-constrained optimization,"
%   Section 3.3 (discretization) and Section 5, Tables 7-8 (heat equation
%   control example with control constraints).
%
%   Problem (continuous):
%
%       min_{y,u}  (1/2) int_0^T int_Omega (y - yhat)^2 dx dt
%                  + (beta/2) int_0^T int_Omega u^2 dx dt
%
%       s.t.       y_t - Lap(y) = u   in  Omega x (0,T],  Omega = [0,1]^2,
%                  y = 0              on  dOmega x (0,T]   (Dirichlet),
%                  y = y0             at  t = 0,
%                  u_a <= u <= u_b.
%
%   Discretization (Pearson-Gondzio Sec. 3.3): Q1 in space (interior n x n
%   grid, h = 1/(n+1)), backward Euler in time (Nt = T/tau steps), trapezoidal
%   rule for the time integrals in the cost.  Stacking all Nt time levels
%   (N = n^2 * Nt unknowns for each of y and u), the discrete QP is
%
%       min (tau/2)(y-yd)'*M12*(y-yd) + (beta*tau/2)*u'*M12*u
%       s.t.  Kst*y - tau*Mst*u = fT,   u_a <= u <= u_b,
%
%   with the space-time operators
%       A_sp = M_sp + tau*K_sp                      (one backward-Euler step)
%       Kst  = blkdiag(A_sp) - subdiag(M_sp)        (block lower-bidiagonal)
%       Mst  = blkdiag(M_sp, ..., M_sp)
%       M12  = blkdiag(M_sp/2, M_sp, ..., M_sp, M_sp/2)   (trapezoid weights)
%       fT   = [M_sp*y0; 0; ...; 0]                  ( = 0 here, y0 = 0 )
%   where K_sp = kron(K1D,M1D)+kron(M1D,K1D) is the Q1 (-Lap) stiffness and
%   M_sp = kron(M1D,M1D) the Q1 mass, exactly as in benchmark_pde_oc_2D_I.
%
%   Desired state (time-independent): yhat = exp(-64*((x1-0.5)^2+(x2-0.5)^2)).
%   This is SMOOTH, so the control active set is clean (no degenerate band as
%   in the discontinuous-target example), and a single tolerance tol_di_eq
%   suffices across the mesh.
%
%   NOTE: These MATLAB files were built with the assistance of Claude
%   (Anthropic) and may contain minor deviations from the paper.
%
%   First pass: tau = 0.04 (Nt = 25), beta = 1e-2, u in [0,1]; sweep
%   h in {2^-2, 2^-3, 2^-4}.  Solvers compared:
%
%       (a) MOSEK / Gurobi / HiGHS / DAQP / OSQP / PIQP  -- interior-point /
%           active-set / first-order references on the explicit space-time QP.
%       (b) di_qp_eq -- full space-time (y,u), default projector onto
%           {Kst*y = tau*Mst*u} (direct Cholesky of Kst*Kst'+tau^2*Mst^2);
%           small-N correctness reference, capped by chol_N_max.
%       (c) di_qp on the null-space-reduced u-only QP: eliminate
%           y = Kst^{-1}(tau*Mst*u + fT) by a forward time march; the reduced
%           Hessian Q_red = tau*G'*M12*G + beta*tau*M12 (G = Kst^{-1}*tau*Mst)
%           is the well-conditioned smoothing operator of the control-bounded
%           case.  Kst^{-1} and Kst^{-T} reuse one Cholesky of A_sp.  Workhorse.
%
% -------------------------------------------------------------------------
%   FUNCTIONS CALLED
%       mosekopt         (MOSEK Optimization Toolbox, interior-point)
%       di_qp_eq         (equality + inequality form, full space-time)
%       di_qp            (inequality form, null-space reduction)
%
%   LOCAL FUNCTIONS
%       build_space_ops  assemble Q1 A_sp, M_sp, K_sp and space-time Kst,Mst,M12
%       make_Asolve      Cholesky of A_sp returned as a solve handle
%       kst_solve        apply Kst^{-1} by a forward backward-Euler time march
%       kst_adj_solve    apply Kst^{-T} by a backward time march
%       mst_apply        apply blkdiag(M_sp) block by block
%       m12_apply        apply trapezoid-weighted blkdiag(M_sp) block by block
%       compute_kkt      KKT residuals for the space-time control-bounded OC
%       store            write one solver's results into res
%       skip_msg         print a uniform skip message
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

h_vals = [2^-5];   % spatial mesh sizes (coarse -> finer)
T      = 1;                    % final time
tau    = 0.02;                 % time step  (Nt = T/tau = 25 intervals)
Nt     = round(T / tau);

% Skip a solver if N = n^2*Nt exceeds the cap below (N is the y-dimension).
mosek_N_max    = 0;
gurobi_N_max   = 0;
highs_N_max    = 0;
daqp_N_max     = 0;     % dense active-set: 2N-vector dense H gets large
osqp_N_max     = 0;
piqp_N_max     = 0;
chol_N_max     = 0;      % di_qp_eq space-time projector Cholesky
nsred_N_max    = Inf;      % NSred di_qp (the workhorse)

% control regularization and bounds (Pearson-Gondzio Table 7, beta = 1e-2 row)
beta = 1e-2;
u_a  = 0;
u_b  = 1;

tol_kkt   = 1e-08;
tol_di_eq = 1e-08;

fprintf('Benchmark: time-dependent heat-equation OC on [0,1]^2, control bounds, Q1 FEM.\n')
fprintf('  PDE  y_t - Lap(y) = u,  homogeneous Dirichlet,  T = %g,  tau = %g (Nt = %d).\n', T, tau, Nt)
fprintf('  beta = %g,  u in [%g, %g]\n', beta, u_a, u_b)
fprintf('  Desired state: yhat = exp(-64*((x1-0.5)^2+(x2-0.5)^2)) (time-independent).\n\n')


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
    'NSred',   true,        nsred_N_max   };

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

nn  = length(h_vals);
pfx = {'mosek','gurobi','highs','daqp','osqp','piqp','chol','nsred'};
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


%% Loop over mesh sizes

for idx = 1:nn

    h = h_vals(idx);
    n = round(1/h) - 1;          % interior grid points per dimension
    Nsp = n^2;                   % spatial DOFs
    N   = Nsp * Nt;              % space-time DOFs (per variable block)

    fprintf('=== h = %g  (n = %d,  Nsp = %d,  Nt = %d,  N = %d) ===\n\n', h, n, Nsp, Nt, N)


    %% Space-time FEM operators (Q1 in space, backward Euler in time)

    [A_sp, M_sp, K_sp, Kst, Mst, M12] = build_space_ops(n, h, tau, Nt);

    % --- Assembly sanity checks ---
    assert(size(Kst,1) == N && size(Mst,1) == N, 'Kst/Mst size mismatch with N.')
    assert(norm(M_sp - M_sp', 'fro') < 1e-10*norm(M_sp,'fro'), 'M_sp not symmetric.')
    assert(all(diag(M_sp) > 0), 'M_sp has nonpositive diagonal.')


    %% Desired state (smooth Gaussian, time-independent)

    x1vec    = (1:n)' * h;
    [X1, X2] = ndgrid(x1vec, x1vec);
    yd_sp    = reshape(exp(-64*((X1-0.5).^2 + (X2-0.5).^2)), [], 1);
    yd       = repmat(yd_sp, Nt, 1);           % stacked over time levels


    %% Per-mesh run flags

    run_mosek  = have_mosek  && (N <= mosek_N_max);
    run_gurobi = have_gurobi && (N <= gurobi_N_max);
    run_highs  = have_highs  && (N <= highs_N_max);
    run_daqp   = have_daqp   && (N <= daqp_N_max);
    run_osqp   = have_osqp   && (N <= osqp_N_max);
    run_piqp   = have_piqp   && (N <= piqp_N_max);
    run_chol   = (N <= chol_N_max);
    run_nsred  = (N <= nsred_N_max);


    %% Explicit QP data (full space-time (y,u) formulation, control bounds)

    H    = blkdiag(tau * M12, beta * tau * M12);
    fvec = [-tau * (M12 * yd); sparse(N, 1)];
    Aeq  = [Kst, -tau * Mst];
    beq  = zeros(N, 1);                          % fT = 0 (y0 = 0, Dirichlet)
    lb   = [-inf(N, 1);  u_a * ones(N, 1)];
    ub   = [ inf(N, 1);  u_b * ones(N, 1)];


    %% Constraint matrices for di_qp_eq (control bounds only)

    E_eq = [Kst, -tau * Mst];                    % N x 2N

    A_di_eq = [sparse(N,N),  speye(N); ...
               sparse(N,N), -speye(N)];           % 2N x 2N
    b_di_eq = [u_b * ones(N,1); -u_a * ones(N,1)];


    %% Feasible starting point: u0 = midpoint, y0 = Kst\(tau*Mst*u0)

    u0 = 0.5*(u_a + u_b) * ones(N, 1);
    Asolve = make_Asolve(A_sp);                  % Cholesky of A_sp as a handle
    y0 = kst_solve(tau * mst_apply(u0, M_sp, Nsp, Nt), Asolve, M_sp, Nsp, Nt);
    x0 = [y0; u0];


    % KKT diagnostics reported for each solver (CONTROL-bounded problem):
    %   kkt_stat  = ||beta*tau*M12*u - tau*Mst*lambda||_inf over interior controls.
    %   kkt_pfeas = max(||Kst*y - tau*Mst*u||_inf, control box violation).
    %   kkt_dfeas = multiplier sign violations, plus -- only for full_diag
    %               solvers -- the state stationarity
    %               ||tau*M12*(y-yd) + Kst'*lambda||_inf (zero by construction
    %               for the DI solvers).


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

        x_mosek    = res_mosek.sol.itr.xx;
        fval_mosek = res_mosek.sol.itr.pobjval;
        flag_mosek = res_mosek.rcode;
        lam_mosek  = -res_mosek.sol.itr.y;   % MOSEK uses opposite sign convention

        y_mosek = x_mosek(1:N);
        u_mosek = x_mosek(N+1:2*N);

        fgap_mosek = res_mosek.sol.itr.pobjval - res_mosek.sol.itr.dobjval;
        [ks, kp, kd] = compute_kkt(y_mosek, u_mosek, lam_mosek, Kst, Mst, M12, yd, beta, tau, u_a, u_b, tol_kkt, true);

        fprintf('  MOSEK:  t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (flag=%d)\n\n', ...
                t_mosek, fval_mosek, ks, kp, kd, fgap_mosek, flag_mosek)

        res = store(res, 'mosek', idx, t_mosek, fval_mosek, ks, kp, kd, fgap_mosek);
    else
        skip_msg('MOSEK', have_mosek, N, mosek_N_max)
    end


    %% 2. Gurobi (interior-point)

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
        rc_u_g = result_g.rc(N+1:2*N);
        fgap_gurobi = result_g.pi' * (beq - Aeq * result_g.x) ...
                    + max(0,  rc_u_g)' * (result_g.x(N+1:2*N) - u_a) ...
                    + max(0, -rc_u_g)' * (u_b - result_g.x(N+1:2*N));
        t_gurobi = toc;

        if ~strcmp(result_g.status, 'OPTIMAL')
            error('Gurobi did not find an optimal solution (status: %s).', result_g.status)
        end

        y_gurobi = result_g.x(1:N);
        u_gurobi = result_g.x(N+1:2*N);
        [ks, kp, kd] = compute_kkt(y_gurobi, u_gurobi, result_g.pi, Kst, Mst, M12, yd, beta, tau, u_a, u_b, tol_kkt, true);

        fprintf('  Gurobi: t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (status=%s)\n\n', ...
                t_gurobi, result_g.objval, ks, kp, kd, fgap_gurobi, result_g.status)

        res = store(res, 'gurobi', idx, t_gurobi, result_g.objval, ks, kp, kd, fgap_gurobi);
    else
        skip_msg('Gurobi', have_gurobi, N, gurobi_N_max)
    end


    %% 3. HiGHS (interior-point)

    if run_highs
        tic
        opts_h.output_flag                  = false;
        opts_h.solver                       = "ipm";
        opts_h.ipm_optimality_tolerance     = 1e-10;
        opts_h.primal_feasibility_tolerance = 1e-10;
        opts_h.dual_feasibility_tolerance   = 1e-10;

        fprintf('  Running HiGHS (IPM)...\n')
        [soln_h, info_h] = callhighs(full(fvec), Aeq, beq, beq, lb, ub, H, [], opts_h);
        rc_u_h = soln_h.col_dual(N+1:2*N);
        lam_h  = -soln_h.row_dual;
        fgap_highs = lam_h' * (beq - Aeq * soln_h.col_value) ...
                   + max(0,  rc_u_h)' * (soln_h.col_value(N+1:2*N) - u_a) ...
                   + max(0, -rc_u_h)' * (u_b - soln_h.col_value(N+1:2*N));
        t_highs = toc;

        if ~strcmp(info_h.model_status_string, 'Optimal')
            error('HiGHS did not find an optimal solution (status: %s).', info_h.model_status_string)
        end

        y_highs = soln_h.col_value(1:N);
        u_highs = soln_h.col_value(N+1:2*N);
        [ks, kp, kd] = compute_kkt(y_highs, u_highs, lam_h, Kst, Mst, M12, yd, beta, tau, u_a, u_b, tol_kkt, true);

        fprintf('  HiGHS:  t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (status=%s)\n\n', ...
                t_highs, info_h.objective_function_value, ks, kp, kd, fgap_highs, info_h.model_status_string)

        res = store(res, 'highs', idx, t_highs, info_h.objective_function_value, ks, kp, kd, fgap_highs);
    else
        skip_msg('HiGHS', have_highs, N, highs_N_max)
    end


    %% 4. DAQP (active-set, dense)

    if run_daqp
        tic
        ms_d     = 2*N;
        bupper_d = [inf(N,1);   u_b*ones(N,1);  full(beq)];
        blower_d = [-inf(N,1);  u_a*ones(N,1);  full(beq)];
        sense_d  = zeros(ms_d + N, 1, 'int32');

        fprintf('  Running DAQP (active-set)...\n')
        [x_d, fval_d, exitflag_d, info_d] = daqp.quadprog( ...
            full(H), full(fvec), full(Aeq), bupper_d, blower_d, sense_d);
        t_daqp = toc;

        if exitflag_d ~= int32(1)
            warning('DAQP did not find an optimal solution (exitflag: %d).', exitflag_d)
        end

        lambda_d   = info_d.lambda;
        lam_eq_d   = lambda_d(ms_d+1:ms_d+N);   % equality adjoint (standard KKT sign)

        fgap_daqp = lam_eq_d' * (beq - Aeq * x_d) ...
                  + max(0, -lambda_d(N+1:2*N))' * (x_d(N+1:2*N) - u_a) ...
                  + max(0,  lambda_d(N+1:2*N))' * (u_b - x_d(N+1:2*N));

        y_d = x_d(1:N);
        u_d = x_d(N+1:2*N);
        [ks, kp, kd] = compute_kkt(y_d, u_d, lam_eq_d, Kst, Mst, M12, yd, beta, tau, u_a, u_b, tol_kkt, true);

        fprintf('  DAQP:   t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (exitflag=%d)\n\n', ...
                t_daqp, fval_d, ks, kp, kd, fgap_daqp, exitflag_d)

        res = store(res, 'daqp', idx, t_daqp, fval_d, ks, kp, kd, fgap_daqp);
    else
        skip_msg('DAQP', have_daqp, N, daqp_N_max)
    end


    %% 5. OSQP (ADMM, first-order)

    if run_osqp
        tic
        A_osqp  = [Aeq; sparse(N,N), speye(N)];
        l_osqp  = [beq;    u_a * ones(N,1)];
        ub_osqp = [beq;    u_b * ones(N,1)];

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

        lam_eq_osqp = res_osqp.y(1:N);
        y_u_osqp    = res_osqp.y(N+1:2*N);
        fgap_osqp = lam_eq_osqp' * (beq - Aeq * res_osqp.x) ...
                  + max(0, -y_u_osqp)' * (res_osqp.x(N+1:2*N) - u_a) ...
                  + max(0,  y_u_osqp)' * (u_b - res_osqp.x(N+1:2*N));

        y_osqp = res_osqp.x(1:N);
        u_osqp = res_osqp.x(N+1:2*N);
        [ks, kp, kd] = compute_kkt(y_osqp, u_osqp, lam_eq_osqp, Kst, Mst, M12, yd, beta, tau, u_a, u_b, tol_kkt, true);

        fprintf('  OSQP:   t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (status=%s)\n\n', ...
                t_osqp, res_osqp.info.obj_val, ks, kp, kd, fgap_osqp, res_osqp.info.status)

        res = store(res, 'osqp', idx, t_osqp, res_osqp.info.obj_val, ks, kp, kd, fgap_osqp);
    else
        skip_msg('OSQP', have_osqp, N, osqp_N_max)
    end


    %% 6. PIQP (proximal interior point)

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

        lam_eq_piqp = result_p.y;
        z_bl_u      = result_p.z_bl(N+1:2*N);
        z_bu_u      = result_p.z_bu(N+1:2*N);
        fgap_piqp = lam_eq_piqp' * (beq - Aeq * result_p.x) ...
                  + z_bl_u' * (result_p.x(N+1:2*N) - u_a) ...
                  + z_bu_u' * (u_b - result_p.x(N+1:2*N));

        y_piqp = result_p.x(1:N);
        u_piqp = result_p.x(N+1:2*N);
        [ks, kp, kd] = compute_kkt(y_piqp, u_piqp, lam_eq_piqp, Kst, Mst, M12, yd, beta, tau, u_a, u_b, tol_kkt, true);

        fprintf('  PIQP:   t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (status=%s)\n\n', ...
                t_piqp, result_p.info.primal_obj, ks, kp, kd, fgap_piqp, result_p.info.status)

        res = store(res, 'piqp', idx, t_piqp, result_p.info.primal_obj, ks, kp, kd, fgap_piqp);
    else
        skip_msg('PIQP', have_piqp, N, piqp_N_max)
    end


    %% 7. di_qp_eq, full space-time (y,u), default projector

    if run_chol
        fprintf('  Running di_qp_eq (full space-time, direct Cholesky)...\n')
        tic
        [x_chol, ~, lam_chol, fval_chol, flag_chol, fgap_chol] = ...
            di_qp_eq(A_di_eq, b_di_eq, fvec, H, E_eq, beq, x0, tol_di_eq, [], [], [], @apgd_lsqnonneg);
        t_chol = toc;

        y_chol = x_chol(1:N);
        u_chol = x_chol(N+1:2*N);
        [ks, kp, kd] = compute_kkt(y_chol, u_chol, lam_chol, Kst, Mst, M12, yd, beta, tau, u_a, u_b, tol_kkt, false);

        fprintf('  Chol:   t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (flag=%d)\n\n', ...
                t_chol, fval_chol, ks, kp, kd, fgap_chol, flag_chol)

        res = store(res, 'chol', idx, t_chol, fval_chol, ks, kp, kd, fgap_chol);
    else
        skip_msg('Chol', true, N, chol_N_max)
    end


    %% 8. Null-space-reduced u-only QP via di_qp (the workhorse)
    %   y = Kst^{-1}(tau*Mst*u + fT) eliminates the equality.  Reduced QP in u:
    %     min (1/2)*u'*Q_red*u + c_red'*u  s.t.  u_a <= u <= u_b,
    %   with G = Kst^{-1}*tau*Mst, Q_red = tau*G'*M12*G + beta*tau*M12, and
    %   c_red = tau*G'*M12*(y_f - yd), y_f = Kst^{-1}*fT.  G uses a forward time
    %   march (kst_solve), G' a backward march (kst_adj_solve), both reusing the
    %   single Cholesky of A_sp.

    if run_nsred
        % Control->state map and its adjoint (matrix-free, reuse A_sp Cholesky).
        Gfun  = @(u) kst_solve(tau * mst_apply(u, M_sp, Nsp, Nt), Asolve, M_sp, Nsp, Nt);
        Gtfun = @(w) tau * mst_apply(kst_adj_solve(w, Asolve, M_sp, Nsp, Nt), M_sp, Nsp, Nt);
        m12f  = @(v) m12_apply(v, M_sp, Nsp, Nt);

        y_f = kst_solve(beq, Asolve, M_sp, Nsp, Nt);     % = 0 here (fT = 0)

        Q_red = @(u) tau * Gtfun(m12f(Gfun(u))) + beta*tau*m12f(u);
        c_red = tau * Gtfun(m12f(y_f - yd));
        A_red = [speye(N); -speye(N)];
        b_red = [u_b*ones(N,1); -u_a*ones(N,1)];

        fprintf('  Running di_qp (null-space reduction)...\n')
        tic
        [u_ns, p_ns, fval_red_ns, flag_ns] = di_qp(A_red, b_red, c_red, ...
            Q_red, u0, tol_di_eq, [], @apgd_lsqnonneg);
        t_nsred = toc;

        y_ns = Gfun(u_ns) + y_f;                          % equality satisfied by construction
        fgap_ns = p_ns' * (b_red - A_red*u_ns);
        lam_ns = -kst_adj_solve(tau * m12f(y_ns - yd), Asolve, M_sp, Nsp, Nt);  % adjoint

        [ks, kp, kd] = compute_kkt(y_ns, u_ns, lam_ns, Kst, Mst, M12, yd, beta, tau, u_a, u_b, tol_kkt, false);

        fprintf('  NSred:  t = %7.3f s,  fval = %+.10e,  kkt_stat = %.2e,  kkt_pfeas = %.2e,  kkt_dfeas = %.2e,  fgap = %.2e  (flag=%d)\n\n', ...
                t_nsred, fval_red_ns, ks, kp, kd, fgap_ns, flag_ns)

        res = store(res, 'nsred', idx, t_nsred, fval_red_ns, ks, kp, kd, fgap_ns);
    else
        skip_msg('NSred', true, N, nsred_N_max)
    end


    %% Store mesh sizes

    res.n(idx) = n;
    res.N(idx) = N;

end  % mesh loop


%% Build data matrices for per-mesh SGM rankings

sgm_solver_names = {'NSred (di_qp)','Chol (di_qp_eq)','MOSEK', ...
                    'Gurobi','HiGHS','OSQP','PIQP','DAQP'};
sgm_field_pfx    = {'nsred','chol','mosek','gurobi','highs','osqp','piqp','daqp'};
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

print_pfx   = {'nsred','chol','mosek','gurobi','highs','osqp','piqp','daqp'};
print_names = {'NSred (di_qp)','Chol (di_qp_eq)','MOSEK', ...
               'Gurobi','HiGHS','OSQP','PIQP','DAQP'};

fprintf('\nSummary\n')

for idx = 1:nn
    fprintf('\n=== h = %g,  N = %d  (tau = %g) ===\n\n', h_vals(idx), res.N(idx), tau)
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

    % --- Per-mesh SGM ranking (ratios relative to best solver at this h) ---
    tc = T_sgm(:, idx);  pc = P_sgm(:, idx);  dc = D_sgm(:, idx);
    sc = S_sgm(:, idx);  gc = G_sgm(:, idx);

    rt = tc ./ min(tc(~isnan(tc)));
    rp = pc ./ min(pc(~isnan(pc)));
    rd = dc ./ min(dc(~isnan(dc)));
    rs = sc ./ min(sc(~isnan(sc)));
    rg = gc ./ min(gc(~isnan(gc)));

    hdr_sgm = sprintf('  %-22s  %12s  %12s  %12s  %12s  %12s', ...
        'Solver', 'Runtime', 'Primal res', 'Dual res', 'Stat. compl.', 'Duality gap');
    fprintf('\n  SGM ratios (relative to best at h = %g):\n', h_vals(idx))
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

function [A_sp, M_sp, K_sp, Kst, Mst, M12] = build_space_ops(n, h, tau, Nt)
%BUILD_SPACE_OPS  Q1 spatial operators on [0,1]^2 and the space-time blocks.
%   K_sp = consistent Q1 (-Lap) stiffness, M_sp = Q1 mass (interior nodes,
%   homogeneous Dirichlet), A_sp = M_sp + tau*K_sp one backward-Euler step.
%   Kst (block lower-bidiagonal), Mst (blkdiag mass), M12 (trapezoid-weighted
%   blkdiag mass) are the Nt-level space-time operators.
    e   = ones(n, 1);
    K1D = spdiags([-e, 2*e, -e], [-1,0,1], n, n) / h;
    M1D = spdiags([ e, 4*e,  e], [-1,0,1], n, n) * (h/6);

    K_sp = kron(K1D, M1D) + kron(M1D, K1D);     % Nsp x Nsp, SPD
    M_sp = kron(M1D, M1D);                       % consistent Q1 mass
    A_sp = M_sp + tau * K_sp;                    % backward-Euler step operator

    I_t   = speye(Nt);
    sub_t = spdiags(ones(Nt,1), -1, Nt, Nt);     % first subdiagonal in time
    Kst   = kron(I_t, A_sp) - kron(sub_t, M_sp); % block lower-bidiagonal
    Mst   = kron(I_t, M_sp);

    theta = ones(Nt, 1);  theta(1) = 0.5;  theta(Nt) = 0.5;   % trapezoid weights
    M12   = kron(spdiags(theta, 0, Nt, Nt), M_sp);
end

function Asolve = make_Asolve(A_sp)
%MAKE_ASOLVE  Cholesky of the SPD step operator A_sp, returned as a solve handle.
    [L, flag, perm] = chol(A_sp, 'lower', 'vector');
    if flag ~= 0, error('chol(A_sp) failed; A_sp = M_sp + tau*K_sp should be SPD.'); end
    Asolve = @(r) asolve_apply(L, perm, r);
end

function x = asolve_apply(L, perm, r)
%ASOLVE_APPLY  Apply A_sp^{-1} to one (or several) right-hand side via Cholesky.
    x = zeros(size(r));
    x(perm, :) = L' \ (L \ r(perm, :));
end

function y = kst_solve(r, Asolve, M_sp, Nsp, Nt)
%KST_SOLVE  Apply Kst^{-1} by a forward backward-Euler time march.
%   Solves Kst*y = r, i.e. A_sp*y_i - M_sp*y_{i-1} = r_i, y_0 = 0.
    R = reshape(r, Nsp, Nt);
    Y = zeros(Nsp, Nt);
    Y(:,1) = Asolve(R(:,1));
    for i = 2:Nt
        Y(:,i) = Asolve(R(:,i) + M_sp * Y(:,i-1));
    end
    y = Y(:);
end

function w = kst_adj_solve(s, Asolve, M_sp, Nsp, Nt)
%KST_ADJ_SOLVE  Apply Kst^{-T} by a backward time march.
%   Solves Kst'*w = s, i.e. A_sp*w_i - M_sp*w_{i+1} = s_i, w_{Nt+1} = 0.
    S = reshape(s, Nsp, Nt);
    W = zeros(Nsp, Nt);
    W(:,Nt) = Asolve(S(:,Nt));
    for i = Nt-1:-1:1
        W(:,i) = Asolve(S(:,i) + M_sp * W(:,i+1));
    end
    w = W(:);
end

function w = mst_apply(v, M_sp, Nsp, Nt)
%MST_APPLY  Apply blkdiag(M_sp,...,M_sp) block by block.
    w = reshape(M_sp * reshape(v, Nsp, Nt), [], 1);
end

function w = m12_apply(v, M_sp, Nsp, Nt)
%M12_APPLY  Apply the trapezoid-weighted blkdiag mass M12 block by block.
    W = M_sp * reshape(v, Nsp, Nt);
    W(:,1)  = 0.5 * W(:,1);
    W(:,Nt) = 0.5 * W(:,Nt);
    w = W(:);
end

function [kkt_stat, kkt_pfeas, kkt_dfeas] = ...
        compute_kkt(y, u, lambda, Kst, Mst, M12, yd, beta, tau, u_a, u_b, tol_kkt, full_diag)
%COMPUTE_KKT  KKT residuals for the space-time control-bounded heat OC.
%
%   lambda is the equality adjoint (Lagrange multiplier of Kst*y - tau*Mst*u = fT).
%   full_diag = true adds the state stationarity ||tau*M12*(y-yd)+Kst'*lambda||_inf
%   to kkt_dfeas (for solvers that do not enforce it by construction); the DI
%   solvers satisfy it by construction (pass false).

    r_u      = beta*tau*(M12*u) - tau*(Mst*lambda);     % control stationarity
    interior = (u > u_a + tol_kkt) & (u < u_b - tol_kkt);
    if any(interior)
        kkt_stat = norm(r_u(interior), inf);
    else
        kkt_stat = 0;
    end

    eq_res    = norm(Kst*y - tau*(Mst*u), inf);          % fT = 0
    ineq_res  = max(0, max([u - u_b; u_a - u]));
    kkt_pfeas = max(eq_res, ineq_res);

    at_ub = u >= u_b - tol_kkt;     % r_u >= 0 expected here
    at_lb = u <= u_a + tol_kkt;     % r_u <= 0 expected here
    kkt_dfeas = max(max([0; -r_u(at_ub)]), max([0; r_u(at_lb)]));

    if full_diag
        state_stat = norm(tau*(M12*(y - yd)) + Kst'*lambda, inf);
        kkt_dfeas  = max(kkt_dfeas, state_stat);
    end
end

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

function str = fmtsgm(v)
%FMTSGM  Format an SGM ratio as a 12-char string, or '---' for NaN.
    if isnan(v), str = sprintf('%12s', '---');
    else,        str = sprintf('%12.3e', v); end
end

function print_row(name, t, vals)
%PRINT_ROW  Print one solver row in the per-h summary table.
%   vals = [primal_res, dual_res, stat_compl, duality_gap].
    fprintf('  %-20s  %11.3f  %12.2e  %12.2e  %12.2e  %12.2e\n', ...
            name, t, vals(1), vals(2), vals(3), vals(4));
end
