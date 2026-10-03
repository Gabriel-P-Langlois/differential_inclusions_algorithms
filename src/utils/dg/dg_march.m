function [F, R] = dg_march(S, K, C, F, tau, nst, solver, m, ref, n_runs)
%DG_MARCH  nst steps of the semi-implicit scheme (21) of Liu-Hu-Taitano-
%   Zhang, each followed by the two-stage postprocessing of section 3.2:
%
%       dx^2 f^n + tau*A f^n = dx^2 f^{n-1} + tau*C f^{n-1},
%
%   with K = decomposition(dx^2*I + tau*A) and C from dg_assemble.  F is the
%   nb-by-Nc coefficient array at the start; solver(w) -> [x, work] solves
%   stage 1 (empty: no limiter); m is the lower bound.  If ref(w) -> x_ref
%   is given, every stage-1 solution is compared with it (dg_limit).  If
%   n_runs is given, each stage-1 solve runs n_runs + 1 times (dg_limit).
%
%   R: per-step rhat (averages below m), work, tlim (solver time, the mean
%   of the timed calls), tall (time of all stage-1 calls of the step), nzs
%   (cells scaled by Zhang-Shu), mass <f_h^n, 1>, mins (minimum average), and err
%   (max-norm distance to ref, only when ref is given); totals nfired,
%   drift = max_n |<f_h^n,1> - <f_h^0,1>| and minavg = min(mins).
%   If a stage-1 solution has an average below m, the march stops at that
%   step: R.stop = the step, R.viol = m - min(x), and the per-step arrays
%   end there (our choice; no clipping).  Otherwise R.stop = 0.
%   The march stops with an error if the solution becomes non-finite.

    if nargin < 9, ref = []; end
    if nargin < 10, n_runs = []; end
    dx = S.dx;  nb = S.nb;  Nc = S.Nc;
    mass0 = dx^2*sum(F(1,:));

    R.rhat = zeros(nst,1);  R.work = zeros(nst,1);  R.tlim = zeros(nst,1);
    R.tall = zeros(nst,1);
    R.nzs  = zeros(nst,1);  R.mass = zeros(nst,1);  R.mins = zeros(nst,1);
    R.stop = 0;  R.viol = 0;
    if ~isempty(ref), R.err = zeros(nst,1); end
    for s = 1:nst
        f = K \ (dx^2*F(:) + tau*(C*F(:)));
        if ~all(isfinite(f))
            error('dg_march: non-finite solution at step %d.', s)
        end
        F = reshape(f, nb, Nc);
        if ~isempty(solver)
            [F, info] = dg_limit(F, S, m, solver, ref, n_runs);
            R.rhat(s) = info.rhat;  R.work(s) = info.work;
            R.tlim(s) = info.time;  R.tall(s) = info.time_all;  R.nzs(s) = info.nzs;
            if ~isempty(ref), R.err(s) = info.err; end
        end
        R.mass(s) = dx^2*sum(F(1,:));
        R.mins(s) = min(F(1,:));
        if ~isempty(solver) && info.below < 0
            R.stop = s;  R.viol = -info.below;
            break
        end
    end
    if R.stop > 0
        for fld = {'rhat', 'work', 'tlim', 'tall', 'nzs', 'mass', 'mins', 'err'}
            if isfield(R, fld{1}), R.(fld{1}) = R.(fld{1})(1:R.stop); end
        end
    end
    R.nfired = nnz(R.rhat);
    R.drift  = max(abs(R.mass - mass0));
    R.minavg = min(R.mins);
end
