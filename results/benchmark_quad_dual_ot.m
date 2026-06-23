%   BENCHMARK_QUAD_DUAL_OT   Regularization-path benchmark for the DUAL of
%                            quadratically-regularized optimal transport,
%                            solved over a whole t-path in one warm-started
%                            call with di_rlp_rpqr and compared against MOSEK.
%
%
%   We consider quadratically-regularized optimal transport between two
%   discrete measures on 2-D point clouds, namely the primal problem:
%   min_P  <C,P> + (s/2)||P||_F^2   s.t.  P 1_n = a,  P^T 1_m = b,  P >= 0.
%
%
%   The unregularized Kantorovich potential LP is
%                       max_y f'y  s.t.  M' y <= c,
%   with the two potentials y = (u,v), f = [a;b], c = vec(C),
%   and the marginal operator M (so (M'y)_{ij} = u_i + v_j).  We
%   regularize the dual problem as follows,
%               max_y  f'y - (t/2)||y||^2   s.t.  M' y <= c
%        <=>    min_y (t/2)||y||^2 - f'y   s.t.  M' y <= c,
%   The dimensions of the dual problem are more favorable than its primal
%   counterpart.
%
%
%   We construct a regularization path in terms of t. The feasible set
%   {M'y <= c} is independent of t, so the optimizer y*(t_k) at one level is
%   feasible and used as a warm start for the next. di_rlp_rpqr marches the
%   ENTIRE path in one call, carrying its QR factorization across levels down
%   to the exact LP at t = 0. As t -> 0 the active set stabilizes, so each
%   level costs only a few breakpoints, whereas an interior-point solver
%   (MOSEK) must re-solve each level from scratch (IPMs do not warm-start
%   across t); we time MOSEK cold at every level as the baseline.
%
%   Interior start:  y0 = 0 is feasible (M'*0 = 0 <= vec(C) since C >= 0); it
%   seeds the path and skips Phase-1.
%
% -------------------------------------------------------------------------


%% Preliminaries
% Put the DI solvers (src/main) and NNLS solvers (src/utils/nnls) on path.
here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', '..', 'src')));


%% Parameters
rng(1);
m       = 200;      % source cloud size
n       = 200;      % target cloud size (n ~= m gives a rectangular P)
t_reg   = 1;        % top of the dual regularization path (Q = t*I)
n_modes = 3;        % Gaussian-mixture components per cloud (multimodal)
tol     = 1e-8;     % outer tolerance for di_rlp_rpqr

% Regularization path, decreasing to 0.
t_path = t_reg * [10:-0.5:0.5, 0.49:-0.01:0.0];


%% Build the OT instance and its dual data
[~, ~, a, b, C] = gen_ot(m, n, n_modes);
mn = m * n;  cvec = C(:);
M = [ kron(ones(1,n), speye(m)) ; kron(speye(n), ones(1,m)) ];
f = [a; b];
npot = m + n;
y0 = zeros(npot, 1);


%% Initialization
have_mosek = exist('mosekopt', 'file') > 0;
fprintf('\n=== Quadratically-regularized OT: dual regularization path ===\n');
fprintf('m = %d   n = %d   n_modes = %d   tol = %g   MOSEK %d\n', ...
        m, n, n_modes, tol, have_mosek);


%% Regularization path di_rlp_rpqr (carried QR factorization)
%   di_rlp_rpqr computes the regularization internally, warm-starting each 
%   level from the previous one and carrying a single (dense) QR 
%   factorization of the active block across breakpoints and levels.
%   It returns the solution at every level as a column of X (dual
%   potentials) and P (plans).
fprintf('\n=== Whole path in one call: di_rlp_rpqr (carried QR) ===\n');
tic;
[Xr, Pr, fvr, flr, bpr] = di_rlp_rpqr(M, cvec, -f, t_path, y0, tol);
t_rpqr = toc;
p_end = Pr(:, end);
fprintf('  levels = %d   total breakpoints = %d   time = %.4f s   converged = %d\n', ...
        numel(t_path), sum(bpr), t_rpqr, all(flr == 1));
if t_path(end) == 0
    fprintf('  t=0 endpoint:  support = %d   dualfeas %.2e   marginal %.2e   <C,P> = %.6e\n', ...
            nnz(p_end > tol), max(0,-min(p_end)), norm(M*p_end - f), cvec' * p_end);
end


%% Per-level MOSEK sweep (cold) for comparison
%   MOSEK cannot warm-start across t (IPMs re-solve each level from scratch), 
%   so we time it cold at every level as the baseline for the whole-path
%   di_rlp_rpqr call above.
fprintf('\n=== Per-level MOSEK sweep (cold):  m = %d, n = %d ===\n', m, n);
fprintf('\n  %-10s %11s\n', 't', 't_mosek(s)');

tot_mo = 0;  fmo = NaN;
for k = 1:numel(t_path)
    tk = t_path(k);
    tmo = NaN;
    if have_mosek
        try
            pc = struct();
            if tk > 0
                pc.qosubi = (1:npot)'; pc.qosubj = (1:npot)'; pc.qoval = tk*ones(npot,1);
            end
            pc.c = -f; pc.a = M.'; pc.blc = -inf(mn,1); pc.buc = cvec;
            pc.blx = -inf(npot,1); pc.bux = inf(npot,1);
            tic; [~, resm] = mosekopt('minimize echo(0)', pc, struct()); tmo = toc;
            fmo = f' * resm.sol.itr.xx;     % dual optimum f'y* = transport cost
        catch err
            fprintf('  MOSEK failed at t=%g: %s\n', tk, err.message);
        end
    end
    if ~isnan(tmo), tot_mo = tot_mo + tmo; end
    if tk == 0, label = '0 (LP)'; else, label = sprintf('%.4g', tk); end
    fprintf('  %-10s %11.4f\n', label, tmo);
end

fprintf('  ---\n');
fprintf('  totals:  di_rlp_rpqr (whole path) %.3f s   MOSEK (cold per level) %.3f s\n', ...
        t_rpqr, tot_mo);
fprintf('  di_rlp_rpqr vs MOSEK = %.2fx\n', tot_mo / t_rpqr);

% t = 0 cross-check: di_rlp_rpqr transport cost against MOSEK's.
if have_mosek && t_path(end) == 0 && ~isnan(fmo)
    fprintf('  t=0 transport cost:  di_rlp_rpqr <C,P> = %.6e   MOSEK = %.6e\n', ...
            cvec' * p_end, fmo);
end


% ===================================================================== %
%  Local functions                                                      %
% ===================================================================== %

function [X, Y, a, b, C] = gen_ot(m, n, n_modes)
%   Synthetic 2-D optimal-transport instance (identical to benchmark_quad_ot.m).
%   Two point clouds from Gaussian mixtures whose n_modes centers are rotated and
%   translated between source and target; non-uniform marginal weights; squared-
%   Euclidean cost normalized to max 1.
ang = pi/5;  R = [cos(ang), -sin(ang); sin(ang), cos(ang)];   % rotation
shift = [1.5; 0.5];                                            % translation
sd = 0.25;                                                     % cluster spread

% Source mixture centers equally spaced on a circle of radius 2; target centers
% are the same points rotated by ang and translated by shift.
th = 2*pi*(0:n_modes-1)'/n_modes;
ctr_s = 2 * [cos(th), sin(th)];
ctr_t = (R * ctr_s')' + shift';

X = zeros(m, 2);  Y = zeros(n, 2);
for i = 1:m
    k = randi(n_modes);  X(i, :) = ctr_s(k, :) + sd * randn(1, 2);
end
for j = 1:n
    k = randi(n_modes);  Y(j, :) = ctr_t(k, :) + sd * randn(1, 2);
end

% Non-uniform empirical weights, bounded away from zero, normalized to unit mass.
a = 0.2 + rand(m, 1);  a = a / sum(a);
b = 0.2 + rand(n, 1);  b = b / sum(b);

% Squared-Euclidean cost, normalized so max(C) = 1.
C = sum(X.^2, 2) + sum(Y.^2, 2)' - 2 * (X * Y');
C = max(C, 0);
C = C / max(C(:));
end
