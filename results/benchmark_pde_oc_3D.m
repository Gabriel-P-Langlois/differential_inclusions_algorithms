%   BENCHMARK_PDE_OC_3D   3-D Poisson optimal control, control bounds,
%                           consistent Q1 mass, mesh sweep.
%
%   Reference: Pearson & Gondzio (2017), Numer. Math. 137:959-999.
%
%   PROBLEM (Pearson & Gondzio 2017, Sec. 5, "3D test problems", Table 6)
%       min_{y,u}  (1/2)*||y - yhat||^2_{L2} + (beta/2)*||u||^2_{L2}
%       s.t.       -Lap(y) = u  in  Omega = [0,1]^3,  y = 0 on dOmega,
%                  u_a <= u <= u_b,
%       with yhat = exp(-64*||x - (0.5,0.5,0.5)||^2).
%
%   FROM THE PAPER
%       The PDE, Omega = [0,1]^3, and yhat (Sec. 5, "3D test problems").
%       Q1 elements for state, control and adjoint (Sec. 5, opening
%       paragraph); consistent, not lumped, mass matrices (Sec. 4.1).
%       Control bounds only, with the two (beta, u_b) pairs of Table 6:
%       beta = 1e-4 with 0 <= u <= 20, and beta = 1e-2 with 0 <= u <= 1;
%       the driver runs the mesh sweep once per pair.  Meshes
%       h = 2^-2, ..., 2^-5 (Table 6).  The sweep stops at h = 2^-5: at
%       h = 2^-6 (n = 63, N = 250047) MOSEK and Gurobi ran out of memory,
%       HiGHS did not finish in 15 minutes, and di_qp took too long.
%
%   ASSUMPTIONS (the paper does not state these for the 3D example)
%       A1  y = 0 on dOmega.  The paper states this for its 2D examples
%           only.
%       A2  Uniform grid on [0,1]^3 with spacing h, so h = 2^-k gives
%           n = 2^k - 1 interior nodes per side: n = 3, 7, 15, 31 for the
%           meshes of Table 6.  This is the h
%           convention of IFISS's 2D grids
%           (grid spacing); IFISS has no 3D code.
%       A3  Desired-state term M_c*(nodal values of yhat), as in IFISS's 2D
%           Poisson-control code (pde_control/square_poissoncontrol.m,
%           Myhat = M*yhat_vec, IFISS 3.6).  The
%           paper's eq. (8) writes the exact integral of yhat against each
%           basis function instead.
%
%   DISCRETIZATION
%       Q1 elements on the grid of A2, N = n^3 interior DOFs.  On a uniform
%       grid the Q1 stiffness is K = K1D (x) M1D (x) M1D + M1D (x) K1D (x) M1D
%       + M1D (x) M1D (x) K1D and the consistent Q1 mass is
%       M_c = M1D (x) M1D (x) M1D.
%
%   SOLVERS
%       (a) MOSEK (interior point), on every mesh.
%       (b) Gurobi (interior point and active set, commercial), on every
%           mesh.
%       (c) HiGHS (interior point, open source), on every mesh.
%       (d) di_qp on the null-space-reduced u-only QP, on every mesh.
%
%       Solver (d) eliminates K*y = M_c*u by y = K^{-1}*M_c*u, leaving
%
%           min_u  (1/2)*u'*Q_red*u + c_red'*u   s.t.  u_a <= u <= u_b,
%
%       with Q_red*u = M_c*(K^{-1}*(M_c*(K^{-1}*(M_c*u)))) + beta*M_c*u and
%       c_red = -M_c*(K^{-1}*(M_c*yd)).  The equality then holds by
%       construction.  Each Q_red application costs four products with M_c
%       and two solves with K, each two triangular solves with L_K.
%
%   FUNCTIONS CALLED
%       di_qp            inequality form, null-space reduction
%       ipm_tolerances   tolerances of MOSEK, Gurobi and HiGHS
%       poisson3d_data   stiffness, mass and desired state (src/utils/pde_oc)
%       poisson3d_reduced  Cholesky of K, Q_red, c_red, state and adjoint
%                        maps for di_qp (src/utils/pde_oc)
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
%       from the shared data (K, M_c, yd, the QP arrays): its format
%       conversion, with H and fvec divided by s = ||fvec||_inf (the
%       objective value and multipliers are multiplied back by s), and the
%       call; for di_qp the factorization of K, c_red,
%       the call, and the recovery of y.  KKT residuals and duality gaps
%       are untimed for every solver.  Each solver runs n_runs + 1 = 6
%       times on every mesh: the first run is an untimed warm-up, the
%       reported time is the average of the other n_runs times, and the
%       diagnostics use the solution of the last run.
%
%   OUTPUT
%       A dated log and a .mat file (res, params) in results/pde_oc_runs/,
%       the .mat file rewritten after every mesh.
%
%   LOCAL FUNCTIONS
%       store_kkt        write one solver's time, fval and residuals to res
%       print_kkt        print one solver's line during the mesh loop
%       print_row        print one solver row of the summary table
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

n_vals       = [3, 7, 15, 31];  % n = 2^k - 1 for h = 2^-2..2^-5
mosek_N_max  = Inf;
gurobi_N_max   = Inf;
highs_N_max    = Inf;

% control regularization and bounds.
pairs = [1e-4, 0, 20; ...
         1e-2, 0,  1];

tol_di  = 1e-08;
tol_ipm = ipm_tolerances();
nnls_solver = @box_lsqnonneg;  % NNLS subproblem of di_qp for this example.
n_runs  = 5;        % timed runs per solver and mesh, after one untimed warm-up; the reported time is their average

% Reference solution: pcg on the free block of the
% final active set of di_qp, then an a posteriori KKT check.
tol_cg   = 1e-14;   % pcg relative residual
maxit_cg = 20000;   % pcg iteration cap (one Q_red product per iteration)
tol_sign = 1e-10;   % gradient sign check on the active set, relative to ||g||_inf


%% Record of the run: dated log and .mat file (written after every mesh)

run_dir = fullfile(script_dir, 'pde_oc_runs');
if ~isfolder(run_dir), mkdir(run_dir); end
stamp    = char(datetime('now', 'Format', 'yyyy-MM-dd_HHmmss'));
mat_file = fullfile(run_dir, sprintf('pde_oc_3D_%s.mat', stamp));
diary(fullfile(run_dir, sprintf('pde_oc_3D_%s.log', stamp)))
params = struct('n_vals', n_vals, 'pairs', pairs, ...
                'tol_di', tol_di, 'tol_ipm', tol_ipm, ...
                'nnls_solver', func2str(nnls_solver), 'tol_cg', tol_cg, ...
                'maxit_cg', maxit_cg, 'tol_sign', tol_sign, 'n_runs', n_runs, ...
                'mosek_N_max', mosek_N_max, 'gurobi_N_max', gurobi_N_max, ...
                'highs_N_max', highs_N_max, 'matlab', version, 'date', stamp);

fprintf('Benchmark: 3-D Poisson OC, control bounds only, consistent Q1 mass, mesh sweep.\n')
for ip = 1:size(pairs, 1)
    fprintf('  pair %d:  beta = %g,  u in [%g, %g]\n', ip, pairs(ip, 1), pairs(ip, 2), pairs(ip, 3))
end
fprintf('  Desired state: Gaussian bump at (0.5, 0.5, 0.5).\n\n')


%% Solver availability

have_mosek  = exist('mosekopt',  'file')  > 0;
have_gurobi = exist('gurobi',    'file')  > 0;
have_highs  = exist('callhighs', 'file')  > 0;

avail_rows = { ...
    'MOSEK',   have_mosek,  mosek_N_max;  ...
    'Gurobi',  have_gurobi, gurobi_N_max; ...
    'HiGHS',   have_highs,  highs_N_max;  ...
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

nn = length(n_vals);
res = struct('n', zeros(nn, 1), 'N', zeros(nn, 1), ...
             'cg_calls', zeros(nn, 1), 'cg_iters', zeros(nn, 1));
% NaN marks a solver that did not run on a mesh.
for kf = {'mosek','gurobi','highs','nsred'}
    p = kf{1};
    res.(['t_'       p]) = nan(nn, 1);
    res.(['fval_'    p]) = nan(nn, 1);
    res.(['r_p_'     p]) = nan(nn, 1);
    res.(['r_d_'     p]) = nan(nn, 1);
    res.(['r_g_'     p]) = nan(nn, 1);
    res.(['r_p_rel_' p]) = nan(nn, 1);
    res.(['r_d_rel_' p]) = nan(nn, 1);
    res.(['r_g_rel_' p]) = nan(nn, 1);
    res.(['status_'  p]) = strings(nn, 1);   % solver status (IPMs)
end
ref_fields = {'ref_pass','ref_n_lb','ref_n_ub','ref_n_free','ref_cg_flag', ...
              'ref_cg_iter','ref_cg_relres','ref_relres_true','ref_slack_min', ...
              'ref_sign_viol','ref_fval','ref_t'};
for k = 1:numel(ref_fields), res.(ref_fields{k}) = nan(nn, 1); end
for k = {'mosek','gurobi','highs','nsred'}
    res.(['err_u_'    k{1}]) = nan(nn, 1);
    res.(['err_y_'    k{1}]) = nan(nn, 1);
    res.(['fgap_ref_' k{1}]) = nan(nn, 1);
end
res.sol = cell(nn, 1);


%% Loop over mesh sizes

for idx = 1:nn

    n = n_vals(idx);
    N = n^3;
    h = 1/(n+1);

    run_mosek   = have_mosek  && (N <= mosek_N_max);
    run_gurobi  = have_gurobi && (N <= gurobi_N_max);
    run_highs   = have_highs  && (N <= highs_N_max);

    fprintf('=== n = %d  (N = %d DOFs,  h = 1/%d) ===\n\n', n, N, n+1)

    sol = struct();   % (u, y) of every solver that runs on this mesh


    %% FEM matrices (Q1 stiffness, consistent Q1 mass) and desired state

    data  = poisson3d_data(n);
    K     = data.K;
    M_mat = data.M;
    yd    = data.yd;


    %% QP data

    H    = blkdiag(M_mat, beta * M_mat);
    fvec = [-M_mat * yd; sparse(N, 1)];
    Aeq  = [K, -M_mat];
    beq  = zeros(N, 1);
    lb   = [-inf(N, 1);  u_a * ones(N, 1)];
    ub   = [ inf(N, 1);  u_b * ones(N, 1)];


    %% Feasible starting point: u0 = midpoint

    u0 = 0.5*(u_a + u_b) * ones(N, 1);


    % Diagnostics (untimed).  The same three residuals score every solver,
    % computed by pde_oc_kkt from that solver's own (x, lambda, z) on the QP
    % packed by pde_oc_qp:
    %
    %   r_p  primal residual, max of the PDE equality residual and the
    %        bound violation;
    %   r_d  dual residual ||P*x + q + A'*lambda + G'*z||_inf, the
    %        stationarity of both the state and the control block;
    %   r_g  duality gap |x'*P*x + q'*x + b'*lambda + h'*z|.
    %
    % r_d and r_g are also reported relative to the size of their terms:
    % the data carry a factor beta*h^3, so absolute residuals read small for
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
            prob_mosek.qosubi   = qi;
            prob_mosek.qosubj   = qj;
            prob_mosek.qoval    = qv;
            prob_mosek.c        = full(fvec / s_ipm);
            prob_mosek.a        = Aeq;
            prob_mosek.blc      = beq;
            prob_mosek.buc      = beq;
            prob_mosek.blx      = lb;
            prob_mosek.bux      = ub;
            param.MSK_IPAR_LOG = 0;
            param.MSK_IPAR_INTPNT_ORDER_METHOD = 'MSK_ORDER_METHOD_FORCE_GRAPHPAR';
            param.MSK_DPAR_INTPNT_QO_TOL_PFEAS   = tol_ipm.mosek_qo;
            param.MSK_DPAR_INTPNT_QO_TOL_DFEAS   = tol_ipm.mosek_qo;
            param.MSK_DPAR_INTPNT_QO_TOL_REL_GAP = tol_ipm.mosek_qo;
            [~, res_mosek] = mosekopt('minimize echo(0)', prob_mosek, param);
            if irun > 0, t_runs(irun) = toc; end
        end
        t_mosek = mean(t_runs);

        x_mosek      = res_mosek.sol.itr.xx;
        fval_mosek   = s_ipm * res_mosek.sol.itr.pobjval;   % back to the unscaled QP
        flag_mosek   = res_mosek.rcode;
        res.status_mosek(idx) = string(res_mosek.sol.itr.solsta);
        lambda_mosek = -s_ipm * res_mosek.sol.itr.y;   % MOSEK uses opposite sign convention

        y_mosek = x_mosek(1:N);
        u_mosek = x_mosek(N+1:2*N);
        sol.mosek = struct('u', u_mosek, 'y', y_mosek);

        % MOSEK reports the two bound multipliers separately; their
        % difference is the reduced cost of the control.
        z_mosek = pde_oc_bound_duals(s_ipm * (res_mosek.sol.itr.slx(qp.iu) ...
                                            - res_mosek.sol.itr.sux(qp.iu)));
        k_mosek = pde_oc_kkt(qp, x_mosek, lambda_mosek, z_mosek);

        print_kkt('MOSEK', t_mosek, fval_mosek, k_mosek, sprintf('flag=%d', flag_mosek))
        res = store_kkt(res, 'mosek', idx, t_mosek, fval_mosek, k_mosek);
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

        x_gurobi      = result_g.x;
        fval_gurobi   = s_ipm * result_g.objval;
        lambda_gurobi = -s_ipm * result_g.pi;

        y_gurobi = x_gurobi(1:N);
        u_gurobi = x_gurobi(N+1:2*N);
        sol.gurobi = struct('u', u_gurobi, 'y', y_gurobi);

        % Gurobi reports one reduced cost per variable, rc = H*x + f + Aeq'*lambda.
        z_gurobi = pde_oc_bound_duals(s_ipm * result_g.rc(qp.iu));
        k_gurobi = pde_oc_kkt(qp, x_gurobi, lambda_gurobi, z_gurobi);

        print_kkt('Gurobi', t_gurobi, fval_gurobi, k_gurobi, ...
                  sprintf('status=%s', result_g.status))
        res = store_kkt(res, 'gurobi', idx, t_gurobi, fval_gurobi, k_gurobi);
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
    %   HiGHS uses (1/2)x'Qx + c'x convention; pass H without the factor 1/2.
    %   soln.row_dual for equality L=U=beq: KKT adjoint mu = -soln.row_dual
    %   (sign flip required; same convention as Gurobi's pi for inequalities).

    if run_highs
        fprintf('  Running HiGHS (IPM)...\n')
        t_runs = zeros(n_runs, 1);
        for irun = 0:n_runs   % irun = 0 is an untimed warm-up
            tic
            s_ipm = norm(fvec, inf);   % objective scale; di_qp divides by ||c_red||_inf
            if s_ipm == 0, s_ipm = 1; end
            opts_h.output_flag                  = false;
            opts_h.solver                       = "ipm";   % active-set cycles on this problem
            opts_h.ipm_optimality_tolerance     = tol_ipm.highs;
            opts_h.primal_feasibility_tolerance = tol_ipm.highs;
            opts_h.dual_feasibility_tolerance   = tol_ipm.highs;
            [soln_h, info_h] = callhighs(full(fvec / s_ipm), Aeq, beq, beq, lb, ub, H / s_ipm, [], opts_h);
            if irun > 0, t_runs(irun) = toc; end
        end
        t_highs = mean(t_runs);   % the residuals below are diagnostics, untimed

        res.status_highs(idx) = string(info_h.model_status_string);   % recorded, as for Gurobi

        x_highs      = soln_h.col_value;
        fval_highs   = s_ipm * info_h.objective_function_value;
        lambda_highs = -s_ipm * soln_h.row_dual;

        y_highs = x_highs(1:N);
        u_highs = x_highs(N+1:2*N);
        sol.highs = struct('u', u_highs, 'y', y_highs);

        % HiGHS reports one dual per column, col_dual = H*x + f + Aeq'*lambda.
        z_highs = pde_oc_bound_duals(s_ipm * soln_h.col_dual(qp.iu));
        k_highs = pde_oc_kkt(qp, x_highs, lambda_highs, z_highs);

        print_kkt('HiGHS', t_highs, fval_highs, k_highs, ...
                  sprintf('status=%s', info_h.model_status_string))
        res = store_kkt(res, 'highs', idx, t_highs, fval_highs, k_highs);
    else
        if ~have_highs
            fprintf('  Skipping HiGHS (not found).\n\n')
        else
            fprintf('  Skipping HiGHS (N = %d > %g).\n\n', N, highs_N_max)
        end
    end


    %% 4. Prefactor K, then solve null-space-reduced u-only QP via di_qp
    %  The timer covers everything this solver needs to return (y, u) from
    %  the shared data: the factorization of K and c_red
    %  (poisson3d_reduced), the call to di_qp, and the recovery of y.
    %  Diagnostics (adjoint, KKT residuals, gap) are outside.

    fprintf('  Running di_qp (null-space reduction)...\n')

    t_runs = zeros(n_runs, 1);
    for irun = 0:n_runs   % irun = 0 is an untimed warm-up
        tic
        % Factor K once (nested dissection ordering) and form c_red; see
        % poisson3d_reduced.  Null-space elimination: y = K^{-1}*M_c*u
        % satisfies K*y = M_c*u.  Reduced QP in u:
        % min (1/2)*u'*Q_red*u + c_red'*u  s.t.  u_a <= u <= u_b.
        red = poisson3d_reduced(data, beta);

        A_red  = [speye(N); -speye(N)];
        b_red  = [u_b*ones(N,1); -u_a*ones(N,1)];

        % red.Q and red.c are normalized by red.s = ||c_red||_inf,
        % so tol_di is relative to the data; the objective value and the
        % multipliers come back in the normalized units and are rescaled.
        [u_cg, p_cg, fval_red_cg, flag_cg] = di_qp(A_red, b_red, red.c, ...
            red.Q, u0, tol_di, [], nnls_solver);
        fval_red_cg = red.s * fval_red_cg;
        p_cg        = red.s * p_cg;

        % Recover state; equality constraint satisfied by construction.
        y_cg = red.G(u_cg);
        if irun > 0, t_runs(irun) = toc; end
    end
    t_cg = mean(t_runs);

    x_cg = [y_cg; u_cg];
    sol.nsred = struct('u', u_cg, 'y', y_cg);

    fval_cg = fval_red_cg;

    % Equality dual from the adjoint equation M_c*(y - yd) + K*mu = 0.
    mu_cg = red.adj(y_cg);

    % di_qp's multipliers already follow the ordering of pde_oc_qp: the rows
    % of A_red = [I; -I] are the upper bounds first, then the lower bounds,
    % and p_cg is nonnegative and back in physical units.
    z_cg = p_cg;

    ub_viol_cg = max(0, max(u_cg - u_b));
    lb_viol_cg = max(0, max(u_a - u_cg));
    fprintf('  NSred diagnostics: min(u)=%.4f, max(u)=%.4f, ub_viol=%.2e, lb_viol=%.2e, n_ub_viol=%d, n_lb_viol=%d\n', ...
            min(u_cg), max(u_cg), ub_viol_cg, lb_viol_cg, ...
            sum(u_cg > u_b), sum(u_cg < u_a));

    k_cg = pde_oc_kkt(qp, x_cg, mu_cg, z_cg);

    print_kkt('NSred', t_cg, fval_cg, k_cg, sprintf('flag=%d', flag_cg))


    %% Store results

    res.n(idx) = n;
    res.N(idx) = N;
    res = store_kkt(res, 'nsred', idx, t_cg, fval_cg, k_cg);


    %% 5. Reference solution and errors of every solver
    %  Fix the final active set of di_qp at its bounds, solve the free
    %  block by pcg to tol_cg, and accept only if the KKT conditions hold a
    %  posteriori (pde_oc_reference).  Untimed.

    fprintf('  Reference: pcg on the free block of the final active set of di_qp...\n')
    tic
    ref = pde_oc_reference(red.Q, red.c, u_cg, p_cg / red.s, u_a, u_b, tol_di, tol_cg, maxit_cg, tol_sign);
    t_ref = toc;
    y_ref = red.G(ref.u);
    ref.fval = red.s * ref.fval;                                  % physical units
    fred  = @(u) red.s * (0.5 * (u' * red.Q(u)) + red.c' * u);   % physical reduced objective

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
    res.sol{idx} = sol;

    res_all{ip} = res;
    save(mat_file, 'res_all', 'params')   % record after every mesh

end  % mesh loop


%% Build data matrices for per-mesh SGM rankings

sgm_solver_names = {'NSred (di_qp)','MOSEK','Gurobi','HiGHS'};
sgm_field_pfx    = {'nsred','mosek','gurobi','highs'};
ns = numel(sgm_solver_names);

T_sgm = nan(ns, nn);
P_sgm = nan(ns, nn);
D_sgm = nan(ns, nn);
G_sgm = nan(ns, nn);
for j = 1 : ns
    pf  = sgm_field_pfx{j};
    tv  = res.(['t_' pf])';                   % 1 × nn
    ok  = ~isnan(tv);
    T_sgm(j, ok) = tv(ok);
    P_sgm(j, ok) = res.(['r_p_rel_' pf])(ok)';
    D_sgm(j, ok) = res.(['r_d_rel_' pf])(ok)';
    G_sgm(j, ok) = res.(['r_g_rel_' pf])(ok)';
end


%% Summary table

hdr = sprintf('  %-20s  %11s  %12s  %12s  %12s  %12s  %12s  %12s', ...
    'Solver', 'Runtime (s)', 'r_p', 'r_p (rel)', 'r_d', 'r_d (rel)', 'r_g', 'r_g (rel)');
sep = ['  ' repmat('-', 1, length(hdr) - 2)];

fprintf('\nSummary, pair %d:  beta = %g,  u in [%g, %g]\n', ip, beta, u_a, u_b)

for idx = 1:nn
    fprintf('\n=== n = %d,  N = %d,  h = 1/%d ===\n\n', ...
            res.n(idx), res.N(idx), res.n(idx) + 1)
    fprintf('%s\n', hdr)
    fprintf('%s\n', sep)

    for j = 1 : ns
        p = sgm_field_pfx{j};
        if ~isnan(res.(['t_' p])(idx))
            print_row(sgm_solver_names{j}, res.(['t_' p])(idx), ...
                [res.(['r_p_' p])(idx),     res.(['r_p_rel_' p])(idx), ...
                 res.(['r_d_' p])(idx),     res.(['r_d_rel_' p])(idx), ...
                 res.(['r_g_' p])(idx),     res.(['r_g_rel_' p])(idx)])
        else
            fprintf('  %-20s  %11s  %12s  %12s  %12s  %12s  %12s  %12s\n', ...
                    [sgm_solver_names{j} ' (skipped)'], '---', '---', '---', '---', '---', '---', '---')
        end
    end

    % --- Per-mesh SGM ranking (ratios relative to best solver at this n) ---
    % For a single problem instance, SGM_s({r}) = r, so the ratio IS the SGM.
    tc = T_sgm(:, idx);  pc = P_sgm(:, idx);
    dc = D_sgm(:, idx);  gc = G_sgm(:, idx);

    rt = tc ./ min(tc(~isnan(tc)));
    rp = pc ./ min(pc(~isnan(pc)));
    rd = dc ./ min(dc(~isnan(dc)));
    rg = gc ./ min(gc(~isnan(gc)));

    hdr_sgm = sprintf('  %-22s  %12s  %12s  %12s  %12s', ...
        'Solver', 'Runtime', 'r_p (rel)', 'r_d (rel)', 'r_g (rel)');
    fprintf('\n  SGM ratios (relative to best at n = %d):\n', res.n(idx))
    fprintf('%s\n  %s\n', hdr_sgm, repmat('-', 1, length(hdr_sgm) - 2))
    for j = 1 : ns
        fprintf('  %-22s  %s  %s  %s  %s\n', sgm_solver_names{j}, ...
            fmtsgm(rt(j)), fmtsgm(rp(j)), fmtsgm(rd(j)), fmtsgm(rg(j)))
    end

    % --- Errors against the reference solution ---
    fprintf('\n  Reference (pass = %d, pcg iters = %d, relres = %.1e, n_lb/n_ub/n_free = %d/%d/%d):\n', ...
            res.ref_pass(idx), res.ref_cg_iter(idx), res.ref_relres_true(idx), ...
            res.ref_n_lb(idx), res.ref_n_ub(idx), res.ref_n_free(idx))
    fprintf('  %-22s  %14s  %14s  %14s\n', 'Solver', '||u-u_ref||_inf', '||y-y_ref||_inf', 'f - f_ref')
    fprintf('  %s\n', repmat('-', 1, 70))
    for j = 1 : ns
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

function str = fmtsgm(v)
%FMTSGM  Format an SGM ratio as a 12-char string, or '---' for NaN.
    if isnan(v), str = sprintf('%12s', '---');
    else,        str = sprintf('%12.3e', v); end
end

function print_row(name, t, vals)
%PRINT_ROW  Print one solver row in the per-n summary table.
%   vals = [r_p, r_p_rel, r_d, r_d_rel, r_g, r_g_rel].
    fprintf('  %-20s  %11.3f  %12.2e  %12.2e  %12.2e  %12.2e  %12.2e  %12.2e\n', ...
            name, t, vals(1), vals(2), vals(3), vals(4), vals(5), vals(6));
end
