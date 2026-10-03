function [F, info] = dg_limit(F, S, m, solver, ref, n_runs)
%DG_LIMIT  Two-stage positivity postprocessing of Liu-Hu-Taitano-Zhang
%   section 3.2 (M = +Inf).
%
%   Stage 1, only if some cell average is below m: solve the cell-average
%   QP (22)/(24) with solver(w) -> [x, work], and replace each cell's average
%   by x, (23).  Stage 2, only if some value at the quadrature points S_E is
%   negative: the Zhang-Shu limiter with epsilon = 0 (our choice;
%   LHTZ leave epsilon unstated), which scales each cell's
%   non-constant modes by theta = min(1, fbar/(fbar - min_q f)).
%
%   info: rhat (averages below m), work (solver's iteration or breakpoint
%   count), time (tic/toc of the solver call), nzs (cells scaled by
%   stage 2), err (||x - ref(w)||_inf, if a reference solver ref is given;
%   computed after the timing, so it does not enter info.time), below
%   (min(x) - m for the
%   solver's output).  If below < 0, stage 2 is skipped: a negative
%   average would make the Zhang-Shu factor negative, and the caller stops
%   the march (our choice; no clipping).
%
%   With n_runs given, stage 1 calls solver(w) n_runs + 1 times on the same
%   w: one untimed warm-up, then n_runs timed calls.  info.time is the mean
%   of the timed calls, info.time_all the time of all n_runs + 1 calls, and
%   x is the output of the last call.  Without n_runs, solver(w) runs once
%   and info.time_all = info.time.

    if nargin < 6, n_runs = []; end
    info = struct('rhat', 0, 'work', 0, 'time', 0, 'time_all', 0, 'nzs', 0, ...
                  'err', 0, 'below', 0);
    w = F(1,:).';
    if any(w < m)
        info.rhat = nnz(w < m);
        if isempty(n_runs)
            t0 = tic;
            [x, info.work] = solver(w);
            info.time = toc(t0);
            info.time_all = info.time;
        else
            t_all = tic;
            solver(w);                                  % untimed warm-up
            t = zeros(n_runs, 1);
            for irun = 1:n_runs
                t0 = tic;
                [x, info.work] = solver(w);
                t(irun) = toc(t0);
            end
            info.time = mean(t);
            info.time_all = toc(t_all);
        end
        info.below = min(x) - m;
        if nargin >= 5 && ~isempty(ref)
            info.err = norm(x - ref(w), inf);
        end
        F(1,:) = x.';
        if info.below < 0, return, end
    end

    mn  = min(S.Phi * F, [], 1);
    bad = mn < 0;
    if any(bad)
        fb = F(1, bad);
        th = min(1, fb ./ (fb - mn(bad)));
        F(2:end, bad) = F(2:end, bad) .* th;
        info.nzs = nnz(bad);
    end
end
