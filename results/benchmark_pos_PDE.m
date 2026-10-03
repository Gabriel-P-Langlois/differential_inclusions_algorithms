%   BENCHMARK_POS_PDE  Positivity-preserving DG scheme of Liu-Hu-Taitano-
%                      Zhang, section 4.1 (linear Fokker-Planck accuracy
%                      test), with the cell-average limiter solved by DR or
%                      by di_pos_w.
%
%   REFERENCE
%       Liu, Hu, Taitano & Zhang, Comput. Math. Appl. 192 (2025) 54-71
%       (LHTZ).
%
%   PROBLEM DATA (source in brackets; our choices marked OURS)
%       Omega = [-10,10]^2, t in [1, 20]                              [4.1]
%       d_t f = Lap f + div(f v), i.e. (17a) with D = I, u = 0, all other
%       coefficients 1, so b = D(u - v) = -v                          [(31), 4.1]
%       exact solution (32); f_h^0 = L2 projection of (32) at t = 1, then
%       Zhang-Shu                                                     [(32), 3.1]
%       P^k, k = 2, 3 (total degree), Legendre orthonormal on
%       [-1/2,1/2]^2, tensor (k+1)-point Gauss                        [3.1]
%       NIPG, sigma = 1, penalty sigma/h with h = dx                  [3.1, Remark 3; OURS]
%       (LHTZ use h two ways: section 3.1 calls h the element diameter,
%       Remark 3 takes h^d = |E|, i.e. h = dx.  OURS: h = dx in
%       both the penalty and the DR norm.)
%       Lax-Friedrichs flux (19), zero on boundary faces (required by
%       a_conv(f,1) = 0 in 3.1); NIPG face terms on interior faces only.
%       NOTE: 4.1 says the boundary data come from the exact solution;
%       the printed forms have no boundary terms, and the exact
%       solution's boundary flux is below 1e-25.
%       semi-implicit scheme (21)                                     [(21)]
%       (tau, dx): Table 1.  P3 row 1: tau = 19/119 (119 steps), since
%       Table 1's 0.16 gives 118.75 steps                             [Table 1; OURS]
%       limiter stage 1: QP (22)/(24), m = 1e-13, M = +Inf, on all
%       cells, when some cell average < m                             [4, 3.2; Remark 4
%                                                                      not applied: OURS]
%       limiter stage 2: Zhang-Shu at S_E, epsilon = 0                [3.2; OURS]
%       DR: Algorithm DR, eps_tol = 1e-13, ||.||_2h with h = dx       [3.3, 4, Remark 3]
%       errors: L2_h and Linf_h at the (k+1)-point Gauss points       [4.1]
%       The conservation row is unweighted (uniform mesh) and M = +Inf, so
%       this experiment exercises neither a weighted row nor two-sided
%       clipping.
%
%   STAGES (list them in `stages` below; they run in that order)
%       3  Table 1 rows with the limiter solved by di_pos_w; item-by-item
%          comparison with Table 1.  Saves results/pos_PDE_runs/*.mat.
%       4  the same rows with the limiter solved by DR; loads the stage-3
%          results and prints DR and di_pos_w side by side.
%       An average below m after a stage-1 solve stops the march at that
%       step (no clipping).
%
%   TIMING
%       Each Table 1 row is marched once.  At every step where stage 1 acts,
%       the solver runs n_runs + 1 = 6 times on the same cell averages: one
%       untimed warm-up, then n_runs timed calls, whose mean is the limiter
%       time of that step; the march continues from the output of the last
%       call.  R.wall is the wall time of the march, and R.wall_one = R.wall
%       minus the time of all stage-1 calls plus the per-step mean times,
%       the wall time of a march with one stage-1 call per step.
%
%   FUNCTIONS CALLED
%       lhtz41_setup, which calls dg_setup, dg_assemble, dg_project and
%       dg_limit, and dg_march (src/utils/dg),
%       limiter_solve (src/utils/limiter), which calls di_pos_w,
%       clip_scale_start, dr_lhtz and dr_params
%
% -------------------------------------------------------------------------


%% Paths

script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', 'src', 'main'));
addpath(fullfile(script_dir, '..', 'src', 'utils', 'dg'));
addpath(fullfile(script_dir, '..', 'src', 'utils', 'limiter'));
addpath(fullfile(script_dir, '..', 'src', 'utils'));
out_dir = fullfile(script_dir, 'pos_PDE_runs');
if ~isfolder(out_dir), mkdir(out_dir); end


%% Parameters

stages    = [3 4];              % run in this order
with_n256 = true;               % add the n = 256 rows of Table 1

n_runs    = 5;                  % timed stage-1 calls per step (averaged), after one untimed warm-up

m_bnd   = 1e-13;                % LHTZ section 4
eps_tol = 1e-13;                % LHTZ section 4

% Table 1: k, n (dx = 20/n), tau, L2_h and Linf_h errors at t_end = 20.
T1 = [2,  64, 0.04,    1.932e-3, 1.666e-3
      2, 128, 0.01,    5.186e-4, 4.626e-4
      2, 256, 0.0025,  1.330e-4, 1.194e-4
      3,  64, 19/119,  1.821e-5, 1.877e-5      % tau: OURS (Table 1: 0.16)
      3, 128, 0.01,    1.188e-6, 1.264e-6
      3, 256, 6.25e-4, 7.555e-8, 8.079e-8];
T1rate = [NaN NaN; 1.898 1.849; 1.963 1.954; NaN NaN; 3.938 3.893; 3.975 3.968];


%% Runs

for stage = stages
    switch stage
        case {3, 4}
            rows = 1:6;
            if ~with_n256, rows(T1(rows,2) == 256) = []; end
            if stage == 3, arm = 'dipos'; else, arm = 'dr'; end
            for r = rows
                k = T1(r,1);  n = T1(r,2);  tau = T1(r,3);
                fprintf('Running k = %d, n = %d, tau = %.6g, limiter %s ...\n', k, n, tau, arm)
                R = run41(k, n, tau, arm, m_bnd, eps_tol, n_runs);
                save(run_file(out_dir, k, n, arm), 'R')
                fprintf('  done in %.1f s\n', R.wall)
            end

            if stage == 3
                fprintf('\nTable 1 comparison (limiter: di_pos_w)\n')
                fprintf('%3s %5s %9s | %11s %11s %6s %6s | %11s %11s %6s %6s\n', 'k', 'n', 'tau', ...
                        'L2 paper', 'L2 ours', 'rate', 'paper', 'Linf paper', 'Linf ours', 'rate', 'paper')
                prev = [NaN NaN];  kprev = 0;
                for r = rows
                    R = load_run(run_file(out_dir, T1(r,1), T1(r,2), 'dipos'));
                    if T1(r,1) ~= kprev, prev = [NaN NaN]; end
                    rt = log2(prev ./ [R.eL2 R.eLinf]);
                    fprintf('%3d %5d %9.6f | %11.4e %11.4e %6.3f %6.3f | %11.4e %11.4e %6.3f %6.3f\n', ...
                            T1(r,1), T1(r,2), T1(r,3), T1(r,4), R.eL2, rt(1), T1rate(r,1), ...
                            T1(r,5), R.eLinf, rt(2), T1rate(r,2))
                    prev = [R.eL2 R.eLinf];  kprev = T1(r,1);
                end
            end

            print_stats(out_dir, T1, rows, m_bnd)
    end
end

%% =========================================================================
%  Local functions
%  =========================================================================

function R = run41(k, n, tau, arm, m, eps_tol, n_runs)
%RUN41  One march of LHTZ section 4.1 from t = 1 to t = 20, with the
%   stage-1 limiter solved by arm = 'dipos' or 'dr'.  Returns the L2_h and
%   Linf_h errors at t = 20 (NaN if the march stopped on an average below
%   m), per-step limiter data, the mass drift max_n |<f_h^n,1> - <f_h^0,1>|,
%   the minimum cell average, and the wall times R.wall and R.wall_one (see
%   TIMING); each stage-1 solve runs n_runs + 1 times.
    t0w = tic;
    nst = round(19/tau);
    assert(abs(nst*tau - 19) < 1e-10, 'tau does not divide [1, 20]')
    [S, K, C, F, f32] = lhtz41_setup(k, n, tau);
    dx = S.dx;

    solver = @(w) limiter_solve(arm, w, m, dx, eps_tol);
    [F, R] = dg_march(S, K, C, F, tau, nst, solver, m, [], n_runs);

    if R.stop > 0
        R.eL2 = NaN;  R.eLinf = NaN;
    else
        E = S.Phi*F - f32(20, S.Xq, S.Yq);
        R.eL2   = sqrt(dx^2*sum(S.wq .* E.^2, 'all'));
        R.eLinf = max(abs(E), [], 'all');
    end
    R.k = k;  R.n = n;  R.tau = tau;  R.arm = arm;  R.nst = nst;
    R.n_runs   = n_runs;
    R.wall     = toc(t0w);
    R.wall_one = R.wall - sum(R.tall) + sum(R.tlim);
end


function f = run_file(out_dir, k, n, arm)
%RUN_FILE  Path of the saved march for degree k, n-by-n cells, limiter arm.
    f = fullfile(out_dir, sprintf('k%d_n%d_%s.mat', k, n, arm));
end


function R = load_run(f)
    L = load(f, 'R');
    R = L.R;
end


function print_stats(out_dir, T1, rows, m)
%PRINT_STATS  Limiter statistics of the saved marches (totals over the
%   march), one line per Table 1 row and arm, then Table 1's errors.
%   stop = step at which an average fell below m (0: none).
    arms = {'dipos', 'dr'};
    fprintf('\nLimiter statistics (totals over the march)\n')
    fprintf('%3s %5s %-7s | %6s %8s %9s %6s %10s %9s %10s %10s | %11s %11s\n', ...
            'k', 'n', 'arm', 'fired', 'rhat', 'work', 'stop', 'lim time', 'wall', ...
            'mass drift', 'min avg-m', 'L2_h', 'Linf_h')
    for r = rows
        for a = 1:numel(arms)
            f = run_file(out_dir, T1(r,1), T1(r,2), arms{a});
            if ~isfile(f), continue, end
            R = load_run(f);
            stop = 0;
            if isfield(R, 'stop'), stop = R.stop; end
            tl = sum(R.tlim);
            wall = R.wall;
            if isfield(R, 'wall_one'), wall = R.wall_one; end
            fprintf('%3d %5d %-7s | %6d %8d %9d %6d %10.3e %9.1f %10.2e %+10.2e | %11.4e %11.4e\n', ...
                    T1(r,1), T1(r,2), arms{a}, R.nfired, sum(R.rhat), sum(R.work), stop, ...
                    tl, wall, R.drift, R.minavg - m, R.eL2, R.eLinf)
        end
        fprintf('%3d %5d %-7s | %75s | %11.4e %11.4e\n', T1(r,1), T1(r,2), 'Table 1', '', ...
                T1(r,4), T1(r,5))
    end
end
