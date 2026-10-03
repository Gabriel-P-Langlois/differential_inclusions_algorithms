%   BENCHMARK_POS_SYNTHETIC  Limiter QP of Liu-Hu-Taitano-Zhang on their
%                            synthetic data (35): timing and accuracy of
%                            DR and di_pos_w.
%
%   REFERENCES
%       Liu, Hu, Taitano & Zhang, Comput. Math. Appl. 192 (2025) 54-71
%       (cited below as LHTZ),
%       section 4.6: equations (24), (30), (35), Remarks 2-3, and Table 3.
%
%       Liu, Buzzard & Zhang, J. Comput. Phys. 519 (2024) 113440 (cited
%       below as LBZ),
%       Appendix A, which LHTZ cite as the source of the synthetic problem.
%
%   PROBLEM
%       LHTZ (24):  min (alpha/2)*||x - w||_2^2  s.t.  1'*x = 1'*w,  m <= x <= M,
%       alpha = 2|E| = 2*dx^2 (LHTZ Lemma 1), m = 1e-13 and M = +Inf (LHTZ
%       section 4, first paragraph).  w holds the point values of LHTZ (35),
%
%           f(x,y) = -0.25                  if -delta/4 + 0.25 <= x <= delta/4 + 0.25,
%                    -0.25                  if -delta/4 + 0.75 <= x <= delta/4 + 0.75,
%                    cos(2*pi*x)^8 + 1e-13  otherwise,
%
%       on a uniform grid of resolution dx on [0,1]^2, dx = 2^-7..2^-11
%       (LHTZ Table 3).
%
%   GRID AND DELTA
%       N = (1/dx)^2: LBZ Appendix A uses "a uniform grid of size 1000^2"
%       and calls it "a problem of size 10^6".
%       OURS, unstated in LHTZ and LBZ:
%         - the points are the lower-left cell corners x_i = i*dx,
%           i = 0..1/dx-1, in both directions;
%         - delta = 0.05.  LHTZ choose delta so that 5% of the point values
%           are negative.  No delta attains 5% exactly on a dyadic grid,
%           and 0.05 is the fraction the two bands cover in the continuum.
%           The realized fraction is printed at every dx.
%       LBZ Appendix A uses the same function as LHTZ (35) but the bound
%       x >= 0, i.e. m = 0; we follow LHTZ, m = 1e-13.
%
%   SOLVERS
%       DR        Algorithm DR of LHTZ (dr_lhtz): iteration (30) from y0 = w,
%                 x0 = S(w), parameters c and lambda from Remark 2
%                 (dr_params), stopped by step 4,
%                 h^(d/2)*||y^{k+1} - y^k||_2 < eps_tol with d = 2 and
%                 h = dx (Remark 3 takes h^d = |E|).  The parameter rule is
%                 timed with the solve.
%       di_pos_w  Algorithm 1, exact and parameter-free, from the
%                 clip-and-scale start (clip_scale_start): the DR start
%                 S(w), scaled toward m so that it conserves mass.  The
%                 start is timed with the solve.
%
%   PROTOCOL
%       Each arm solves the same instance n_rep times after one discarded
%       warm-up call; tic/toc measures the n_rep solves and the reported
%       time is their mean, the quantity LHTZ Table 3 reports.
%
%   OUTPUT
%       Per dx, the instance and one row per method.  Then two tables in the
%       layout of LHTZ Table 3, one column per dx and one row per method:
%         1. mean time of one solve, with LHTZ Table 3's DR time per solve
%            in a separate row (a different machine);
%         2. accuracy: ||x - x_ref||_inf against the bisection reference
%            limiter_exact_w, and the conservation residual |1'x - 1'w|
%            (the reference's own residual included), followed by the
%            largest bound violation over all methods.
%
%       A dated log and a .mat file (res, params) go to
%       results/pos_synthetic_runs/, the .mat file rewritten after every dx.
%
%   FUNCTIONS CALLED
%       di_pos_w, clip_scale_start, dr_lhtz, dr_params, limiter_exact_w
%
%   LOCAL FUNCTIONS
%       f35              the synthetic profile, LHTZ (35)
%       print_rows       one row per method of the across-dx tables
%
% -------------------------------------------------------------------------


%% Paths

script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', 'src', 'main'));
addpath(fullfile(script_dir, '..', 'src', 'utils', 'limiter'));
addpath(fullfile(script_dir, '..', 'src', 'utils'));


%% Parameters

dx_vals  = 2.^-(7:11);          % LHTZ Table 3
delta    = 0.05;                % OURS
m_bnd    = 1e-13;               % LHTZ section 4: lower bound m
M_bnd    = inf;                 % LHTZ section 4: upper bound M = +Inf
eps_tol  = 1e-13;               % LHTZ section 4: tolerance of Algorithm DR
n_rep    = 100;                 % timed solves per arm, after one untimed warm-up (LHTZ: 100)
dr_maxit = 100000;
tol_di   = 1e-10;               % di_pos_w feasibility tolerance

% LHTZ Table 3: average CPU time (s) of Algorithm DR for one solve, Intel
% Xeon CPU E5-2660 v3 2.60 GHz.
t3_dx   = 2.^-(7:11);
t3_time = [9.048e-3, 3.068e-2, 1.029e-1, 5.979e-1, 2.705e0];

fprintf('Benchmark: limiter QP (24) of LHTZ on the synthetic data (35).\n')
fprintf('  dx = [%s],  delta = %g (ours),  lower-left cell corners (ours)\n', ...
        num2str(dx_vals), delta)
fprintf('  bounds [%g, %g],  eps_tol = %g,  %d timed solves per arm after one warm-up\n\n', ...
        m_bnd, M_bnd, eps_tol, n_rep)


%% Record of the run: dated log and .mat file (written after every dx)

run_dir = fullfile(script_dir, 'pos_synthetic_runs');
if ~isfolder(run_dir), mkdir(run_dir); end
stamp    = char(datetime('now', 'Format', 'yyyy-MM-dd_HHmmss'));
mat_file = fullfile(run_dir, sprintf('pos_synthetic_%s.mat', stamp));
diary(fullfile(run_dir, sprintf('pos_synthetic_%s.log', stamp)))
params = struct('dx_vals', dx_vals, 'delta', delta, 'm_bnd', m_bnd, ...
                'M_bnd', M_bnd, 'eps_tol', eps_tol, 'n_rep', n_rep, ...
                'dr_maxit', dr_maxit, 'tol_di', tol_di, ...
                'matlab', version, 'date', stamp);


%% Result storage

nd   = numel(dx_vals);
arms = {'dr', 'dipos'};
arm_label = {'DR', 'di_pos_w'};
res = struct();
res.dx = dx_vals(:);
res.N  = zeros(nd,1);   res.fneg = zeros(nd,1);   res.rhat = zeros(nd,1);
res.theta = zeros(nd,1);  res.c = zeros(nd,1);    res.lam  = zeros(nd,1);
res.nact  = zeros(nd,1);  res.cons_ref = nan(nd,1);   % |1'x_ref - 1'w|
for a = 1:numel(arms)
    res.(['time_' arms{a}]) = nan(nd,1);    % mean time of one solve over the n_rep timed solves (s)
    res.(['work_' arms{a}]) = nan(nd,1);    % iterations (DR) or breakpoints (di_pos_w) of one solve
    res.(['err_'  arms{a}]) = nan(nd,1);    % ||x - x_ref||_inf
    res.(['cons_' arms{a}]) = nan(nd,1);    % |1'x - 1'w|
    res.(['pmin_' arms{a}]) = nan(nd,1);    % min(x) - m
    res.(['nbad_' arms{a}]) = zeros(nd,1);  % DR at maxit, or di_pos_w flag ~= 1
end


%% Solves, one instance per dx

for idx = 1:nd

    dx = dx_vals(idx);
    n  = round(1/dx);
    xg = (0:n-1)' * dx;             % lower-left cell corners (OURS)
    [X, ~] = meshgrid(xg, xg);      % f does not depend on y
    w  = f35(X(:), delta);
    N  = n^2;
    one   = ones(N, 1);
    b     = sum(w);

    [c_dr, lam_dr, th] = dr_params(w, m_bnd, M_bnd);        % LHTZ Remark 2
    x_ref = limiter_exact_w(w, m_bnd, M_bnd, one);

    res.N(idx) = N;  res.fneg(idx) = mean(w < 0);  res.rhat(idx) = nnz(w < m_bnd);
    res.theta(idx) = th;  res.c(idx) = c_dr;  res.lam(idx) = lam_dr;
    res.nact(idx) = nnz(x_ref <= m_bnd);  res.cons_ref(idx) = abs(sum(x_ref) - b);

    fprintf('=== dx = 2^%d  (N = %d) ===\n', round(log2(dx)), N)
    fprintf('  Instance: %.4f%% negative point values (delta = %g), rhat = %d, theta_hat = %.4f rad, c = %.4f, lambda = %.4f\n', ...
            100*res.fneg(idx), delta, res.rhat(idx), th, c_dr, lam_dr)
    fprintf('  Reference: |1''x_ref - 1''w| = %.2e, min(x_ref) - m = %.2e, %d of %d points on the lower bound\n', ...
            res.cons_ref(idx), min(x_ref) - m_bnd, res.nact(idx), N)

    % Arm 1: DR, the parameter rule timed with the solve.
    [c_dr, lam_dr] = dr_params(w, m_bnd, M_bnd);                    % warm-up
    dr_lhtz(w, m_bnd, M_bnd, c_dr, lam_dr, eps_tol, dr_maxit, dx);
    t0 = tic;
    for r = 1:n_rep
        [c_dr, lam_dr] = dr_params(w, m_bnd, M_bnd);
        [x, its] = dr_lhtz(w, m_bnd, M_bnd, c_dr, lam_dr, eps_tol, dr_maxit, dx);
    end
    res.time_dr(idx) = toc(t0) / n_rep;
    res.work_dr(idx) = its;
    res.nbad_dr(idx) = (its == dr_maxit);
    res.err_dr(idx)  = norm(x - x_ref, inf);
    res.cons_dr(idx) = abs(sum(x) - b);
    res.pmin_dr(idx) = min(x) - m_bnd;

    % Arm 2: di_pos_w, the clip-and-scale start timed with the solve.
    x0 = clip_scale_start(w, m_bnd, M_bnd, one);                    % warm-up
    di_pos_w(w, m_bnd, M_bnd, one, x0, tol_di);
    t0 = tic;
    for r = 1:n_rep
        x0 = clip_scale_start(w, m_bnd, M_bnd, one);
        [x, flag, nbp] = di_pos_w(w, m_bnd, M_bnd, one, x0, tol_di);
    end
    res.time_dipos(idx) = toc(t0) / n_rep;
    res.work_dipos(idx) = nbp;
    res.nbad_dipos(idx) = (flag ~= 1);
    res.err_dipos(idx)  = norm(x - x_ref, inf);
    res.cons_dipos(idx) = abs(sum(x) - b);
    res.pmin_dipos(idx) = min(x) - m_bnd;

    fprintf('  %-10s  %14s  %8s  %12s  %12s  %12s  %s\n', 'Arm', ...
            'mean time (s)', 'its/solve', 'err vs ref', '|1''x - 1''w|', 'min(x) - m', 'notes')
    fprintf('  %s\n', repmat('-', 1, 106))
    for a = 1:numel(arms)
        nm = arms{a};
        switch nm
            case 'dr',    note = sprintf('iterations; %d at maxit', res.nbad_dr(idx));
            case 'dipos', note = sprintf('breakpoints; flag %d', flag);
        end
        fprintf('  %-10s  %14.4e  %8d  %12.2e  %12.2e  %+12.2e  %s\n', arm_label{a}, ...
                res.(['time_' nm])(idx), res.(['work_' nm])(idx), ...
                res.(['err_' nm])(idx), res.(['cons_' nm])(idx), res.(['pmin_' nm])(idx), note)
    end
    k3 = find(abs(t3_dx - dx) < eps(dx), 1);
    if ~isempty(k3)
        fprintf('  LHTZ Table 3, Algorithm DR: %.3e s per solve (Xeon E5-2660 v3, a different machine)\n', ...
                t3_time(k3))
    end
    fprintf('\n')

    save(mat_file, 'res', 'params')   % record after every dx
end


%% Tables across dx, in the layout of LHTZ Table 3

hdr  = sprintf('  %-14s', 'dx');
Nrow = sprintf('  %-14s', 'N');
for idx = 1:nd
    hdr  = [hdr  sprintf('  %11s', sprintf('2^%d', round(log2(res.dx(idx)))))]; %#ok<AGROW>
    Nrow = [Nrow sprintf('  %11d', res.N(idx))];                                %#ok<AGROW>
end
rule = ['  ' repmat('-', 1, 14 + 13*nd)];

fprintf('=== Table 1: mean time (s) of one solve over %d timed solves, tic/toc ===\n', n_rep)
fprintf('%s\n%s\n%s\n', hdr, Nrow, rule)
print_rows(res, arms, arm_label, 'time_', '%11.3e')
row = sprintf('  %-14s', 'LHTZ T3');
for idx = 1:nd
    k3 = find(abs(t3_dx - res.dx(idx)) < eps(res.dx(idx)), 1);
    if isempty(k3)
        row = [row sprintf('  %11s', '---')];         %#ok<AGROW>
    else
        row = [row sprintf('  %11.3e', t3_time(k3))]; %#ok<AGROW>
    end
end
fprintf('%s\n', row)
fprintf('  (LHTZ Table 3: average time of one solve of Algorithm DR on an Intel Xeon E5-2660 v3, a different machine.)\n\n')

fprintf('=== Table 2: accuracy ===\n')
fprintf('  ||x - x_ref||_inf, x_ref from limiter_exact_w (bisection, independent of the solvers)\n')
fprintf('%s\n%s\n%s\n', hdr, Nrow, rule)
print_rows(res, arms, arm_label, 'err_', '%11.2e')
fprintf('\n  Conservation residual |1''x - 1''w|\n')
fprintf('%s\n%s\n%s\n', hdr, Nrow, rule)
print_rows(res, arms, arm_label, 'cons_', '%11.2e')
print_rows(res, {'ref'}, {'reference'}, 'cons_', '%11.2e')
bv = 0;
for a = 1:numel(arms)
    bv = max([bv; -res.(['pmin_' arms{a}])]);
end
fprintf('\n  Largest bound violation max(0, m - min(x)) over all methods and dx: %.2e\n', bv)

save(mat_file, 'res', 'params')
fprintf('\nResults saved to %s\n', mat_file)
diary off


%% =========================================================================
%  Local functions
%  =========================================================================

function f = f35(x, delta)
%F35  The synthetic profile (35) of LHTZ at abscissae x; it does not
%   depend on y.  The two bands take -0.25, the rest cos(2*pi*x)^8 + 1e-13.
    f = cos(2*pi*x).^8 + 1e-13;
    band = (-delta/4 + 0.25 <= x & x <= delta/4 + 0.25) | ...
           (-delta/4 + 0.75 <= x & x <= delta/4 + 0.75);
    f(band) = -0.25;
end


function print_rows(res, arms, labels, pfx, fmt)
%PRINT_ROWS  One row per method of the across-dx tables: the field
%   res.(pfx arm) printed with format fmt, '---' where the method did not
%   run.
    for a = 1:numel(arms)
        v = res.([pfx arms{a}]);
        row = sprintf('  %-14s', labels{a});
        for idx = 1:numel(v)
            if isnan(v(idx))
                row = [row sprintf('  %11s', '---')];          %#ok<AGROW>
            else
                row = [row sprintf(['  ' fmt], v(idx))];       %#ok<AGROW>
            end
        end
        fprintf('%s\n', row)
    end
end
