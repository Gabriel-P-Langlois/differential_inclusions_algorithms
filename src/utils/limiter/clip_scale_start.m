function x0 = clip_scale_start(u, m, M, w)
%CLIP_SCALE_START  Feasible start for di_pos_w built from the clip
%   S(u) = min(max(u, m), M), the start of Algorithm DR in Liu-Hu-Taitano-
%   Zhang.  The clip violates conservation; one scaling toward the lower
%   bound restores it:
%
%       x0 = m + s*(S(u) - m),   s = (w'*u - m*sum(w)) / (w'*S(u) - m*sum(w)),
%
%   so that w'*x0 = w'*u and m <= x0 <= S(u) <= M.  This needs 0 <= s <= 1:
%   the clip raises the weighted sum (lower-bound violations dominate, as in
%   the positivity limiter) and the weighted mean of u is at least m.
%   Otherwise the function errors.
%
%   One pass over the data.  It does not solve the QP: the minimizer is a
%   shift of the data, clip(u + nu), and x0 is a scaling of the clip.
    xc = min(max(u, m), M);
    ms = m*sum(w);
    s  = (w'*u - ms) / (w'*xc - ms);
    if ~(s >= 0 && s <= 1)
        error('clip_scale_start: scale factor s = %g is outside [0, 1].', s)
    end
    x0 = m + s*(xc - m);
end
