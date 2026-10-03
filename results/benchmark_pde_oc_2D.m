%   BENCHMARK_PDE_OC_2D  Time-dependent distributed control of the heat
%                           equation, control bounds, Q1 in space and
%                           backward Euler in time.
%
%   Reference: Pearson & Gondzio (2017), Numer. Math. 137:959-999, Sec. 3.3
%   and Sec. 5, Tables 7-8.
%
%   PROBLEM
%       min_{y,u}  (1/2)*int_0^T ||y - yhat||^2 dt
%                  + (beta/2)*int_0^T ||u||^2 dt
%       s.t.       y_t - Lap(y) = u  in  Omega x (0,T],  Omega = [0,1]^2,  T = 1,
%                  y = 0 on dOmega,  y = y0 = 0 at t = 0,  u_a <= u <= u_b,
%       with yhat(x,t) = A_yhat*t*sin(2*pi*x1*x2).
%
%   FROM THE PAPER (Pearson & Gondzio 2017)
%       The PDE y_t - Lap(y) = u for t in (0,1] with control bounds only
%       (Sec. 5, "Time-dependent PDE constraints").  Q1 elements for state,
%       control and adjoint (Sec. 5, opening paragraph); consistent, not
%       lumped, mass matrices (Sec. 4.1).  Backward Euler in time and the
%       trapezoidal rule in the cost, giving the block lower-bidiagonal Kst
%       and M12 = blkdiag(M/2, M, ..., M, M/2) below (Sec. 3.3).  The
%       two (beta, u_b) pairs of Table 7, beta = 1e-2 with 0 <= u <= 1 and
%       beta = 1e-4 with 0 <= u <= 30; the driver runs the mesh sweep once
%       per pair.  Time steps tau = 0.04, 0.02, 0.01 and meshes
%       h = 2^-2, ..., 2^-6 (Table 7); this driver uses tau = 0.02.
%
%   ASSUMPTIONS (the paper states none of these for its heat example)
%       A1  Omega = [0,1]^2, T = 1.  The paper states neither the domain nor
%           its dimension for this example.
%       A2  y = 0 on dOmega and y0 = 0, as in the heat-control examples of
%           Stoll & Wathen (2010, Secs. 2 and 6) and Pearson, Stoll & Wathen
%           (2012, Sec. 4), which Pearson & Gondzio cite.
%       A3  yhat(x,t) = A_yhat*t*sin(2*pi*x1*x2): the 2D form of Stoll &
%           Wathen's yhat = t*sin(2*pi*x1*x2*x3) (2010, Sec. 6.2.2).  It
%           changes sign on [0,1]^2 (yhat < 0 where x1*x2 > 1/2), so both
%           control bounds can bind.  A_yhat = 0.5 was chosen so that both
%           bounds are active.  Measured with HiGHS at
%           tau = 0.02, beta = 1e-2: at h = 2^-3 and 2^-4, 67 of 2450 and
%           643 of 11250 controls sit at u = 0 and 400 and 1591 at u = 1; at
%           h = 2^-2 the lower bound is inactive (0 at u = 0, 81 of 450 at
%           u = 1).  The driver uses the same tau = 0.02.
%       A4  y_{d,i} = M_sp*(nodal values of yhat(., t_i)) at t_i = i*tau,
%           i = 1..Nt, as in IFISS's 2D Poisson-control code
%           (pde_control/square_poissoncontrol.m, Myhat = M*yhat_vec).  The
%           paper says only that y_{d,i} relates to the values of yhat at
%           the i-th time step (Sec. 3.3).
%       A5  h is the grid spacing on [0,1]^2, so h = 2^-k gives n = 2^k - 1
%           interior nodes per side (IFISS's 2D grid convention).
%
%   DISCRETIZATION
%       Q1 on an interior n x n grid (h = 1/(n+1)), backward Euler over Nt
%       steps, trapezoidal weights in time.  Stacking the time levels
%       (N = n^2*Nt unknowns for each of y and u),
%
%           min (tau/2)(y-yd)'*M12*(y-yd) + (beta*tau/2)*u'*M12*u
%           s.t. Kst*y - tau*Mst*u = fT,   u_a <= u <= u_b,
%
%       with A_sp = M_sp + tau*K_sp one backward-Euler step, Kst its block
%       lower-bidiagonal stack, Mst = blkdiag(M_sp), M12 the trapezoid-
%       weighted stack, and fT = [M_sp*y0; 0; ...], zero here.  On the
%       uniform interior grid K_sp = kron(K1D, M1D) + kron(M1D, K1D) and
%       M_sp = kron(M1D, M1D), with K1D and M1D the 1-D Q1 stiffness and
%       consistent mass; see heat2d_data.
%
%   SOLVERS
%       (a) MOSEK and HiGHS on the explicit space-time QP, on every mesh.
%           Gurobi on the same QP for h = 2^-2..2^-5 only (gurobi_N_max =
%           31^2*Nt): at h = 2^-6 it did not converge within four hours.
%       (b) di_qp on the null-space-reduced u-only QP: eliminate
%           y = Kst^{-1}(tau*Mst*u + fT), leaving
%           Q_red = tau*G'*M12*G + beta*tau*M12 with G = Kst^{-1}*tau*Mst.
%           On the uniform grid of A1 and A5, K1D and M1D
%           share the orthogonal DST-I eigenvectors S, so K_sp, M_sp and
%           A_sp are diagonal in the 2D sine basis.  G, G' and M12 are
%           applied there: one 2D sine transform per time level, then one
%           scalar backward-Euler recurrence per spatial mode.  This is exact
%           on this grid; it does not carry over to a general mesh.
%
%   FUNCTIONS CALLED
%       di_qp            inequality form, null-space reduction
%       ipm_tolerances   tolerances of MOSEK, Gurobi and HiGHS
%       heat2d_data      shared operators and desired state (src/utils/pde_oc)
%       heat2d_reduced   sine basis, Q_red, c_red, state and adjoint maps
%                        for di_qp (src/utils/pde_oc)
%       box_lsqnonneg    closed-form NNLS subproblem of di_qp on the box
%       pde_oc_reference reference solution from the final active set of
%                        di_qp, checked a posteriori
%       pde_oc_qp        pack the full QP as (P, q, A, b, G, h)
%       pde_oc_kkt       primal residual, dual residual and duality gap of
%                        one solver's (x, lambda, z)
%       pde_oc_bound_duals  split a reduced cost into bound multipliers
%       mosekopt         MOSEK Optimization Toolbox   (optional)
%       gurobi           Gurobi MATLAB interface      (optional)
%       callhighs        HiGHS MEX interface          (optional)
%
%   TIMING
%       Each solver's time covers all the work it needs to return (y, u)
%       from the shared data (Kst, Mst, M12, yd, the QP arrays): its format
%       conversion, with H and fvec divided by s = ||fvec||_inf (the
%       objective value and multipliers are multiplied back by s), and the
%       call; for di_qp the sine-basis setup, c_red, the
%       call, and the recovery of y.  KKT residuals and duality gaps are
%       untimed for every solver.  Each solver runs n_runs + 1 = 6 times
%       on every mesh: the first run is an untimed warm-up, the reported
%       time is the average of the other n_runs times, and the diagnostics
%       use the solution of the last run.
%
%   OUTPUT
%       A dated log and a .mat file (res, params) in results/pde_oc_runs/,
%       the .mat file rewritten after every mesh.
%
%   LOCAL FUNCTIONS
%       store_kkt        write one solver's time, fval and residuals to res
%       print_kkt        print one solver's line during the mesh loop
%       skip_msg         print a uniform skip message
%       print_row        print one solver row of the summary table
%       sgm_ratio        ratios to the best solver at one mesh, NaN if none ran
%       fmtsgm           format an SGM ratio for the ranking table
%
% -------------------------------------------------------------------------


%% Paths

script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', 'src', 'main'));
addpath(fullfile(script_dir, '..', 'src', 'utils', 'nnls'));
addpath(fullfile(script_dir, '..', 'src', 'utils'));
addpath(fullfile(script_dir, '..', 'src', 'utils', 'pde_oc'));

% Optional solvers.  Edit these paths to match your installation.  A solver
% that is neither in its folder nor on the MATLAB path is skipped.
p_mosek  = fullfile(getenv('HOME'), 'mosek', '11.2', 'toolbox', 'r2022b');
p_gurobi = '/Library/gurobi1302/macos_universal2/matlab';
p_highs  = fullfile(getenv('HOME'), 'HiGHSMEX');

if isfolder(p_mosek),  addpath(p_mosek);  end
if isfolder(p_gurobi), addpath(p_gurobi); end
if isfolder(p_highs),  addpath(p_highs);  end


%% Parameters

h_vals = 2.^-(2:6);               % spatial mesh sizes, n = 1/h - 1.
T      = 1;                    % final time
tau    = 0.02;                 % time step  (Nt = T/tau = 50 intervals)
Nt     = round(T / tau);

% Skip a solver if N = n^2*Nt exceeds the cap below (N is the y-dimension).
mosek_N_max    = Inf;
gurobi_N_max   = 48050;        % 31^2*Nt, i.e. h >= 2^-5
highs_N_max    = Inf;
nsred_N_max    = Inf;

% control regularization and bounds: the two (beta, u_a, u_b) pairs of
% Pearson & Gondzio 2017, Table 7.  The whole mesh sweep runs once per
% pair.
pairs = [1e-2, 0,  1; ...
         1e-4, 0, 30];

A_yhat = 0.5;   % amplitude of the desired state (assumption A3)

tol_di    = 1e-08;
tol_ipm   = ipm_tolerances();   % tolerances of MOSEK, Gurobi, HiGHS
nnls_solver = @box_lsqnonneg;   % NNLS subproblem of di_qp: closed form on the box
n_runs    = 5;      % timed runs per solver and mesh, after one untimed warm-up; the reported time is their average

% Reference solution: pcg on the free block of the
% final active set of di_qp, then an a posteriori KKT check.
tol_cg   = 1e-14;   % pcg relative residual
maxit_cg = 20000;   % pcg iteration cap (one Q_red product per iteration)
tol_sign = 1e-10;   % gradient sign check on the active set, relative to ||g||_inf


%% Record of the run: dated log and .mat file (written after every mesh)

run_dir = fullfile(script_dir, 'pde_oc_runs');
if ~isfolder(run_dir), mkdir(run_dir); end
stamp    = char(datetime('now', 'Format', 'yyyy-MM-dd_HHmmss'));
mat_file = fullfile(run_dir, sprintf('pde_oc_2D_%s.mat', stamp));
diary(fullfile(run_dir, sprintf('pde_oc_2D_%s.log', stamp)))
params = struct('h_vals', h_vals, 'T', T, 'tau', tau, 'Nt', Nt, ...
                'pairs', pairs, 'A_yhat', A_yhat, ...
                'tol_di', tol_di, 'tol_ipm', tol_ipm, ...
                'nnls_solver', func2str(nnls_solver), 'tol_cg', tol_cg, ...
                'maxit_cg', maxit_cg, 'tol_sign', tol_sign, 'n_runs', n_runs, ...
                'mosek_N_max', mosek_N_max, 'gurobi_N_max', gurobi_N_max, ...
                'highs_N_max', highs_N_max, 'nsred_N_max', nsred_N_max, ...
                'matlab', version, 'date', stamp);

fprintf('Benchmark: time-dependent heat-equation OC on [0,1]^2, control bounds, Q1 FEM.\n')
fprintf('  PDE  y_t - Lap(y) = u,  homogeneous Dirichlet,  T = %g,  tau = %g (Nt = %d).\n', T, tau, Nt)
for ip = 1:size(pairs, 1)
    fprintf('  pair %d:  beta = %g,  u in [%g, %g]\n', ip, pairs(ip, 1), pairs(ip, 2), pairs(ip, 3))
end
fprintf('  Desired state: yhat = %g*t*sin(2*pi*x1*x2).\n\n', A_yhat)


%% Solver availability

have_mosek  = exist('mosekopt',  'file')  > 0;
have_gurobi = exist('gurobi',    'file')  > 0;
have_highs  = exist('callhighs', 'file')  > 0;

avail_rows = { ...
    'MOSEK',   have_mosek,  mosek_N_max;  ...
    'Gurobi',  have_gurobi, gurobi_N_max; ...
    'HiGHS',   have_highs,  highs_N_max;  ...
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


%% Loop over the (beta, bounds) pairs.  Everything below, through the
%  summary table, runs once per pair; res_all{ip} holds the res of pair ip.

npairs  = size(pairs, 1);
res_all = cell(npairs, 1);
for ip = 1:npairs

beta = pairs(ip, 1);
u_a  = pairs(ip, 2);
u_b  = pairs(ip, 3);
fprintf('\n################  pair %d of %d:  beta = %g,  u in [%g, %g]  ################\n\n', ...
        ip, npairs, beta, u_a, u_b)


%% Result storage

nn  = length(h_vals);
pfx = {'mosek','gurobi','highs','nsred'};
res = struct('n', zeros(nn,1), 'N', zeros(nn,1));
for k = 1:numel(pfx)
    p = pfx{k};
    res.(['t_'         p]) = nan(nn,1);
    res.(['fval_'      p]) = nan(nn,1);
    res.(['r_p_'       p]) = nan(nn,1);
    res.(['r_d_'       p]) = nan(nn,1);
    res.(['r_g_'       p]) = nan(nn,1);
    res.(['r_p_rel_'   p]) = nan(nn,1);
    res.(['r_d_rel_'   p]) = nan(nn,1);
    res.(['r_g_rel_'   p]) = nan(nn,1);
    res.(['status_'    p]) = strings(nn,1);   % solver status (IPMs)
    res.(['err_u_'     p]) = nan(nn,1);   % against the reference
    res.(['err_y_'     p]) = nan(nn,1);
    res.(['fgap_ref_'  p]) = nan(nn,1);
end
ref_fields = {'ref_pass','ref_n_lb','ref_n_ub','ref_n_free','ref_cg_flag', ...
              'ref_cg_iter','ref_cg_relres','ref_relres_true','ref_slack_min', ...
              'ref_sign_viol','ref_fval','ref_t'};
for k = 1:numel(ref_fields), res.(ref_fields{k}) = nan(nn, 1); end
res.sol = cell(nn, 1);


%% Loop over mesh sizes

for idx = 1:nn

    h = h_vals(idx);


    %% Space-time FEM operators (Q1 in space, backward Euler in time) and
    %  the desired state yhat(x,t) = A_yhat*t*sin(2*pi*x1*x2) (assumption A3)

    data = heat2d_data(h, tau, T, A_yhat);
    n    = data.n;                % interior grid points per dimension
    Nsp  = data.Nsp;              % spatial DOFs
    N    = data.N;                % space-time DOFs (per variable block)
    M_sp = data.M_sp;
    Kst  = data.Kst;
    Mst  = data.Mst;
    M12  = data.M12;
    yd   = data.yd;               % levels t_i = i*tau, i = 1..Nt

    fprintf('=== h = %g  (n = %d,  Nsp = %d,  Nt = %d,  N = %d) ===\n\n', h, n, Nsp, Nt, N)

    sol = struct();   % (u, y) of every solver that runs on this mesh

    % --- Assembly sanity checks ---
    assert(size(Kst,1) == N && size(Mst,1) == N, 'Kst/Mst size mismatch with N.')
    assert(norm(M_sp - M_sp', 'fro') < 1e-10*norm(M_sp,'fro'), 'M_sp not symmetric.')
    assert(all(diag(M_sp) > 0), 'M_sp has nonpositive diagonal.')


    %% Per-mesh run flags

    run_mosek  = have_mosek  && (N <= mosek_N_max);
    run_gurobi = have_gurobi && (N <= gurobi_N_max);
    run_highs  = have_highs  && (N <= highs_N_max);
    run_nsred  = (N <= nsred_N_max);


    %% Explicit QP data (full space-time (y,u) formulation, control bounds)

    H    = blkdiag(tau * M12, beta * tau * M12);
    fvec = [-tau * (M12 * yd); sparse(N, 1)];
    Aeq  = [Kst, -tau * Mst];
    beq  = zeros(N, 1);                          % fT = 0 (y0 = 0, Dirichlet)
    lb   = [-inf(N, 1);  u_a * ones(N, 1)];
    ub   = [ inf(N, 1);  u_b * ones(N, 1)];


    %% Feasible starting point: u0 = midpoint

    u0 = 0.5*(u_a + u_b) * ones(N, 1);


    % Diagnostics (untimed).  The same three residuals score every solver,
    % computed by pde_oc_kkt from that solver's own (x, lambda, z) on the QP
    % packed by pde_oc_qp:
    %
    %   r_p  primal residual, max of the state-equation residual and the
    %        control box violation;
    %   r_d  dual residual ||P*x + q + A'*lambda + G'*z||_inf, the
    %        stationarity of both the state and the control block;
    %   r_g  duality gap |x'*P*x + q'*x + b'*lambda + h'*z|.
    %
    % r_d and r_g are also reported relative to the size of their terms: the
    % data carry a factor beta*tau*h^2, so absolute residuals read small for
    % every solver.

    qp = pde_oc_qp(H, fvec, Aeq, beq, u_a, u_b, N);


    %% 1. MOSEK

    if run_mosek
        fprintf('  Running MOSEK...\n')
        t_runs = zeros(n_runs, 1);
        for irun = 0:n_runs   % irun = 0 is an untimed warm-up
            tic
            s_ipm = norm(fvec, inf);   % objective scale; di_qp divides by ||c_red||_inf
            if s_ipm == 0, s_ipm = 1; end
            [qi, qj, qv] = find(tril(H / s_ipm));
            prob_mosek.qosubi = qi;
            prob_mosek.qosubj = qj;
            prob_mosek.qoval  = qv;
            prob_mosek.c      = full(fvec / s_ipm);
            prob_mosek.a      = Aeq;
            prob_mosek.blc    = beq;
            prob_mosek.buc    = beq;
            prob_mosek.blx    = lb;
            prob_mosek.bux    = ub;
            param.MSK_IPAR_LOG = 0;
            param.MSK_DPAR_INTPNT_QO_TOL_PFEAS   = tol_ipm.mosek_qo;
            param.MSK_DPAR_INTPNT_QO_TOL_DFEAS   = tol_ipm.mosek_qo;
            param.MSK_DPAR_INTPNT_QO_TOL_REL_GAP = tol_ipm.mosek_qo;
            [~, res_mosek] = mosekopt('minimize echo(0)', prob_mosek, param);
            if irun > 0, t_runs(irun) = toc; end
        end
        t_mosek = mean(t_runs);

        x_mosek    = res_mosek.sol.itr.xx;
        fval_mosek = s_ipm * res_mosek.sol.itr.pobjval;   % back to the unscaled QP
        flag_mosek = res_mosek.rcode;
        res.status_mosek(idx) = string(res_mosek.sol.itr.solsta);
        lam_mosek  = -s_ipm * res_mosek.sol.itr.y;   % MOSEK uses opposite sign convention

        y_mosek = x_mosek(1:N);
        u_mosek = x_mosek(N+1:2*N);
        sol.mosek = struct('u', u_mosek, 'y', y_mosek);

        % MOSEK reports the two bound multipliers separately; their
        % difference is the reduced cost of the control.
        z_mosek = pde_oc_bound_duals(s_ipm * (res_mosek.sol.itr.slx(qp.iu) ...
                                            - res_mosek.sol.itr.sux(qp.iu)));
        k_mosek = pde_oc_kkt(qp, x_mosek, lam_mosek, z_mosek);

        print_kkt('MOSEK', t_mosek, fval_mosek, k_mosek, sprintf('flag=%d', flag_mosek))
        res = store_kkt(res, 'mosek', idx, t_mosek, fval_mosek, k_mosek);
    else
        skip_msg('MOSEK', have_mosek, N, mosek_N_max)
    end


    %% 2. Gurobi (interior-point)
    %   Gurobi's stationarity reads H*x + f - Aeq'*pi = rc, so mu = -result.pi.

    if run_gurobi
        fprintf('  Running Gurobi...\n')
        t_runs = zeros(n_runs, 1);
        for irun = 0:n_runs   % irun = 0 is an untimed warm-up
            tic
            s_ipm = norm(fvec, inf);   % objective scale; di_qp divides by ||c_red||_inf
            if s_ipm == 0, s_ipm = 1; end
            model_g.Q     = H / (2 * s_ipm);
            model_g.obj   = full(fvec / s_ipm);
            model_g.A     = Aeq;
            model_g.rhs   = full(beq);
            model_g.sense = repmat('=', N, 1);
            model_g.lb    = full(lb);
            model_g.ub    = full(ub);
            params_g.OutputFlag     = 0;
            params_g.BarConvTol     = tol_ipm.gurobi_bar;
            params_g.FeasibilityTol = tol_ipm.gurobi_fo;
            params_g.OptimalityTol  = tol_ipm.gurobi_fo;
            result_g = gurobi(model_g, params_g);
            if irun > 0, t_runs(irun) = toc; end
        end
        t_gurobi = mean(t_runs);   % the residuals below are diagnostics, untimed

        % A non-optimal status keeps the returned point, scored like any
        % other; the status is recorded.
        res.status_gurobi(idx) = string(result_g.status);

        y_gurobi = result_g.x(1:N);
        u_gurobi = result_g.x(N+1:2*N);
        sol.gurobi = struct('u', u_gurobi, 'y', y_gurobi);

        % Gurobi reports one reduced cost per variable, rc = H*x + f + Aeq'*lambda.
        z_gurobi = pde_oc_bound_duals(s_ipm * result_g.rc(qp.iu));
        k_gurobi = pde_oc_kkt(qp, result_g.x, -s_ipm * result_g.pi, z_gurobi);

        print_kkt('Gurobi', t_gurobi, s_ipm * result_g.objval, k_gurobi, ...
                  sprintf('status=%s', result_g.status))
        res = store_kkt(res, 'gurobi', idx, t_gurobi, s_ipm * result_g.objval, k_gurobi);
    else
        skip_msg('Gurobi', have_gurobi, N, gurobi_N_max)
    end


    %% 3. HiGHS (interior-point)

    if run_highs
        fprintf('  Running HiGHS (IPM)...\n')
        t_runs = zeros(n_runs, 1);
        for irun = 0:n_runs   % irun = 0 is an untimed warm-up
            tic
            s_ipm = norm(fvec, inf);   % objective scale; di_qp divides by ||c_red||_inf
            if s_ipm == 0, s_ipm = 1; end
            opts_h.output_flag                  = false;
            opts_h.solver                       = "ipm";
            opts_h.ipm_optimality_tolerance     = tol_ipm.highs;
            opts_h.primal_feasibility_tolerance = tol_ipm.highs;
            opts_h.dual_feasibility_tolerance   = tol_ipm.highs;
            [soln_h, info_h] = callhighs(full(fvec / s_ipm), Aeq, beq, beq, lb, ub, H / s_ipm, [], opts_h);
            if irun > 0, t_runs(irun) = toc; end
        end
        t_highs = mean(t_runs);   % the residuals below are diagnostics, untimed
        lam_h   = -s_ipm * soln_h.row_dual;

        res.status_highs(idx) = string(info_h.model_status_string);   % recorded, as for Gurobi

        y_highs = soln_h.col_value(1:N);
        u_highs = soln_h.col_value(N+1:2*N);
        sol.highs = struct('u', u_highs, 'y', y_highs);

        % HiGHS reports one dual per column, col_dual = H*x + f + Aeq'*lambda.
        z_highs = pde_oc_bound_duals(s_ipm * soln_h.col_dual(qp.iu));
        k_highs = pde_oc_kkt(qp, soln_h.col_value, lam_h, z_highs);

        print_kkt('HiGHS', t_highs, s_ipm * info_h.objective_function_value, k_highs, ...
                  sprintf('status=%s', info_h.model_status_string))
        res = store_kkt(res, 'highs', idx, t_highs, s_ipm * info_h.objective_function_value, k_highs);
    else
        skip_msg('HiGHS', have_highs, N, highs_N_max)
    end


    %% 4. Null-space-reduced u-only QP via di_qp
    %   y = Kst^{-1}(tau*Mst*u + fT) eliminates the equality.  Reduced QP in u:
    %     min (1/2)*u'*Q_red*u + c_red'*u  s.t.  u_a <= u <= u_b,
    %   with G = Kst^{-1}*tau*Mst, Q_red = tau*G'*M12*G + beta*tau*M12, and
    %   c_red = tau*G'*M12*(y_f - yd), y_f = Kst^{-1}*fT.  All operators act in
    %   the 2D sine basis, where M_sp = diag(mu) and A_sp = diag(den), so Kst^{-1}
    %   and Kst^{-T} are scalar recurrences per mode; each Q_red product costs
    %   two 2D sine transforms of the stacked array.

    %   The timer covers the sine-basis setup and c_red (heat2d_reduced),
    %   the call to di_qp, and the recovery of y.  Diagnostics (adjoint,
    %   KKT residuals, gap) are outside.

    if run_nsred
        fprintf('  Running di_qp (null-space reduction)...\n')
        t_runs = zeros(n_runs, 1);
        for irun = 0:n_runs   % irun = 0 is an untimed warm-up
            tic
            red   = heat2d_reduced(data, beta);    % sine basis, Q_red, c_red, G, adjoint
            A_red = [speye(N); -speye(N)];
            b_red = [u_b*ones(N,1); -u_a*ones(N,1)];

            % red.Q and red.c are normalized by red.s = ||c_red||_inf, so
            % tol_di is relative to the data; the objective value and
            % the multipliers come back in the normalized units and are
            % rescaled here.
            [u_ns, p_ns, fval_red_ns, flag_ns] = di_qp(A_red, b_red, red.c, ...
                red.Q, u0, tol_di, [], nnls_solver);
            fval_red_ns = red.s * fval_red_ns;
            p_ns        = red.s * p_ns;

            y_ns = red.G(u_ns) + red.y_f;                 % equality satisfied by construction
            if irun > 0, t_runs(irun) = toc; end
        end
        t_nsred = mean(t_runs);

        sol.nsred = struct('u', u_ns, 'y', y_ns);
        lam_ns = red.adj(y_ns);                           % adjoint

        % di_qp's multipliers already follow the ordering of pde_oc_qp: the
        % rows of A_red = [I; -I] are the upper bounds first, then the lower
        % bounds, and p_ns is nonnegative and back in physical units.
        k_nsred = pde_oc_kkt(qp, [y_ns; u_ns], lam_ns, p_ns);

        print_kkt('NSred', t_nsred, fval_red_ns, k_nsred, sprintf('flag=%d', flag_ns))
        res = store_kkt(res, 'nsred', idx, t_nsred, fval_red_ns, k_nsred);
    else
        skip_msg('NSred', true, N, nsred_N_max)
    end


    %% Store mesh sizes

    res.n(idx) = n;
    res.N(idx) = N;


    %% Reference solution and errors of every solver
    %  Fix the final active set of di_qp at its bounds, solve the free
    %  block by pcg to tol_cg, and accept only if the KKT conditions hold a
    %  posteriori (pde_oc_reference).  Untimed.

    if run_nsred
        fprintf('  Reference: pcg on the free block of the final active set of di_qp...\n')
        tic
        ref = pde_oc_reference(red.Q, red.c, u_ns, p_ns / red.s, u_a, u_b, tol_di, tol_cg, maxit_cg, tol_sign);
        t_ref = toc;
        y_ref = red.G(ref.u) + red.y_f;
        ref.fval = red.s * ref.fval;                                  % physical units
        fred  = @(u) red.s * (0.5 * (u' * red.Q(u)) + red.c' * u);   % physical reduced objective (up to a constant)

        fprintf(['  Reference: pass = %d,  n_lb = %d,  n_ub = %d,  n_free = %d,  ' ...
                 'pcg flag = %d, iters = %d, relres = %.2e (true %.2e),  ' ...
                 'slack_min = %.2e,  sign_viol = %.2e,  t = %.2f s\n'], ...
                ref.pass, ref.n_lb, ref.n_ub, ref.n_free, ref.cg_flag, ref.cg_iter, ...
                ref.cg_relres, ref.relres_true, ref.slack_min, ref.sign_viol, t_ref)

        res.ref_pass(idx)        = ref.pass;
        res.ref_n_lb(idx)        = ref.n_lb;
        res.ref_n_ub(idx)        = ref.n_ub;
        res.ref_n_free(idx)      = ref.n_free;
        res.ref_cg_flag(idx)     = ref.cg_flag;
        res.ref_cg_iter(idx)     = ref.cg_iter;
        res.ref_cg_relres(idx)   = ref.cg_relres;
        res.ref_relres_true(idx) = ref.relres_true;
        res.ref_slack_min(idx)   = ref.slack_min;
        res.ref_sign_viol(idx)   = ref.sign_viol;
        res.ref_fval(idx)        = ref.fval;
        res.ref_t(idx)           = t_ref;

        sol.ref = struct('u', ref.u, 'y', y_ref);
        names = fieldnames(sol);
        for j = 1:numel(names)
            s = names{j};
            if strcmp(s, 'ref'), continue, end
            res.(['err_u_' s])(idx)    = norm(sol.(s).u - ref.u, inf);
            res.(['err_y_' s])(idx)    = norm(sol.(s).y - y_ref, inf);
            res.(['fgap_ref_' s])(idx) = fred(sol.(s).u) - ref.fval;
            fprintf('  %-8s vs reference:  ||u - u_ref||_inf = %.2e,  ||y - y_ref||_inf = %.2e,  f - f_ref = %+.2e\n', ...
                    s, res.(['err_u_' s])(idx), res.(['err_y_' s])(idx), res.(['fgap_ref_' s])(idx))
        end
        fprintf('\n')
    end
    res.sol{idx} = sol;

    res_all{ip} = res;
    save(mat_file, 'res_all', 'params')   % record after every mesh

end  % mesh loop


%% Build data matrices for per-mesh SGM rankings

sgm_solver_names = {'NSred (di_qp)','MOSEK','Gurobi','HiGHS'};
sgm_field_pfx    = {'nsred','mosek','gurobi','highs'};
ns_sgm = numel(sgm_solver_names);

T_sgm = nan(ns_sgm, nn);
P_sgm = nan(ns_sgm, nn);
D_sgm = nan(ns_sgm, nn);
G_sgm = nan(ns_sgm, nn);
for j = 1 : ns_sgm
    pf = sgm_field_pfx{j};
    tv = res.(['t_' pf])';
    ok = ~isnan(tv);
    T_sgm(j, ok) = tv(ok);
    P_sgm(j, ok) = res.(['r_p_rel_' pf])(ok)';
    D_sgm(j, ok) = res.(['r_d_rel_' pf])(ok)';
    G_sgm(j, ok) = res.(['r_g_rel_' pf])(ok)';
end


%% Summary table

hdr = sprintf('  %-20s  %11s  %12s  %12s  %12s  %12s  %12s  %12s', ...
    'Solver', 'Runtime (s)', 'r_p', 'r_p (rel)', 'r_d', 'r_d (rel)', 'r_g', 'r_g (rel)');
sep = ['  ' repmat('-', 1, length(hdr) - 2)];

print_pfx   = {'nsred','mosek','gurobi','highs'};
print_names = {'NSred (di_qp)','MOSEK','Gurobi','HiGHS'};

fprintf('\nSummary, pair %d:  beta = %g,  u in [%g, %g]\n', ip, beta, u_a, u_b)

for idx = 1:nn
    fprintf('\n=== h = %g,  N = %d  (tau = %g) ===\n\n', h_vals(idx), res.N(idx), tau)
    fprintf('%s\n', hdr)
    fprintf('%s\n', sep)

    for j = 1 : numel(print_pfx)
        p = print_pfx{j};
        if ~isnan(res.(['t_' p])(idx))
            print_row(print_names{j}, res.(['t_' p])(idx), ...
                [res.(['r_p_' p])(idx),     res.(['r_p_rel_' p])(idx), ...
                 res.(['r_d_' p])(idx),     res.(['r_d_rel_' p])(idx), ...
                 res.(['r_g_' p])(idx),     res.(['r_g_rel_' p])(idx)])
        else
            fprintf('  %-20s  %11s  %12s  %12s  %12s  %12s  %12s  %12s\n', ...
                    [print_names{j} ' (skipped)'], '---', '---', '---', '---', '---', '---', '---')
        end
    end

    % --- Per-mesh SGM ranking (ratios relative to best solver at this h) ---
    tc = T_sgm(:, idx);  pc = P_sgm(:, idx);
    dc = D_sgm(:, idx);  gc = G_sgm(:, idx);

    rt = sgm_ratio(tc);
    rp = sgm_ratio(pc);
    rd = sgm_ratio(dc);
    rg = sgm_ratio(gc);

    hdr_sgm = sprintf('  %-22s  %12s  %12s  %12s  %12s', ...
        'Solver', 'Runtime', 'r_p (rel)', 'r_d (rel)', 'r_g (rel)');
    fprintf('\n  SGM ratios (relative to best at h = %g):\n', h_vals(idx))
    fprintf('%s\n  %s\n', hdr_sgm, repmat('-', 1, length(hdr_sgm) - 2))
    for j = 1 : ns_sgm
        fprintf('  %-22s  %s  %s  %s  %s\n', sgm_solver_names{j}, ...
            fmtsgm(rt(j)), fmtsgm(rp(j)), fmtsgm(rd(j)), fmtsgm(rg(j)))
    end

    % --- Errors against the reference solution ---
    fprintf('\n  Reference (pass = %d, pcg iters = %d, relres = %.1e, n_lb/n_ub/n_free = %d/%d/%d):\n', ...
            res.ref_pass(idx), res.ref_cg_iter(idx), res.ref_relres_true(idx), ...
            res.ref_n_lb(idx), res.ref_n_ub(idx), res.ref_n_free(idx))
    fprintf('  %-22s  %14s  %14s  %14s\n', 'Solver', '||u-u_ref||_inf', '||y-y_ref||_inf', 'f - f_ref')
    fprintf('  %s\n', repmat('-', 1, 70))
    for j = 1 : ns_sgm
        pf = sgm_field_pfx{j};
        fprintf('  %-22s  %s  %s  %s\n', sgm_solver_names{j}, ...
            fmtsgm(res.(['err_u_' pf])(idx)), fmtsgm(res.(['err_y_' pf])(idx)), ...
            fmtsgm(res.(['fgap_ref_' pf])(idx)))
    end
    fprintf('\n\n')
end

res_all{ip} = res;
end  % pair loop

save(mat_file, 'res_all', 'params')
fprintf('Results saved to %s\n', mat_file)
diary off


%% =========================================================================
%  Local functions
%  =========================================================================

function res = store_kkt(res, pfx, idx, t, fval, k)
%STORE_KKT  Write one solver's time, objective value and residuals into res.
    res.(['t_'       pfx])(idx) = t;
    res.(['fval_'    pfx])(idx) = fval;
    res.(['r_p_'     pfx])(idx) = k.r_p;
    res.(['r_d_'     pfx])(idx) = k.r_d;
    res.(['r_g_'     pfx])(idx) = k.r_g;
    res.(['r_p_rel_' pfx])(idx) = k.r_p_rel;
    res.(['r_d_rel_' pfx])(idx) = k.r_d_rel;
    res.(['r_g_rel_' pfx])(idx) = k.r_g_rel;
end

function print_kkt(name, t, fval, k, status)
%PRINT_KKT  Print one solver's line during the mesh loop.
    fprintf(['  %-6s  t = %7.3f s,  fval = %+.10e,  r_p = %.2e (rel %.2e),  ' ...
             'r_d = %.2e (rel %.2e),  r_g = %.2e (rel %.2e),  z_min = %.2e  (%s)\n\n'], ...
            name, t, fval, k.r_p, k.r_p_rel, k.r_d, k.r_d_rel, k.r_g, k.r_g_rel, ...
            k.z_min, status);
end

function skip_msg(name, have, N, nlim)
%SKIP_MSG  Print a uniform skip message for an unavailable / oversized solver.
    if ~have
        fprintf('  Skipping %s (not found).\n\n', name)
    else
        fprintf('  Skipping %s (N = %d > %g).\n\n', name, N, nlim)
    end
end

function r = sgm_ratio(v)
%SGM_RATIO  Ratios of a column to its smallest finite entry; all-NaN if none.
    if all(isnan(v)), r = nan(size(v));
    else,             r = v ./ min(v(~isnan(v))); end
end

function str = fmtsgm(v)
%FMTSGM  Format an SGM ratio as a 12-char string, or '---' for NaN.
    if isnan(v), str = sprintf('%12s', '---');
    else,        str = sprintf('%12.3e', v); end
end

function print_row(name, t, vals)
%PRINT_ROW  Print one solver row in the per-h summary table.
%   vals = [r_p, r_p_rel, r_d, r_d_rel, r_g, r_g_rel].
    fprintf('  %-20s  %11.3f  %12.2e  %12.2e  %12.2e  %12.2e  %12.2e  %12.2e\n', ...
            name, t, vals(1), vals(2), vals(3), vals(4), vals(5), vals(6));
end
