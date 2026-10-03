function [x, work] = limiter_solve(name, w, m, h, eps_tol)
%LIMITER_SOLVE  One solve of the cell-average limiter QP (24) of Liu-Hu-
%   Taitano-Zhang with M = +Inf:
%       min ||x - w||^2  s.t.  sum(x) = sum(w),  x >= m.
%
%   name = 'dipos': di_pos_w from the clip-and-scale start; work = breakpoints.
%   name = 'dr':    Algorithm DR (dr_lhtz) with the Remark 2 parameters
%                   (dr_params), stopped by step 4 with ||.||_2h = h*||.||_2
%                   and tolerance eps_tol; work = iterations.
%   The start and the parameter rule are part of the solve.

    switch name
        case 'dipos'
            one = ones(numel(w), 1);
            x0 = clip_scale_start(w, m, inf, one);
            [x, flag, work] = di_pos_w(w, m, inf, one, x0, 1e-10);
            if flag ~= 1
                error('limiter_solve: di_pos_w returned flag %d.', flag)
            end
        case 'dr'
            [c, lam] = dr_params(w, m, inf);
            [x, work] = dr_lhtz(w, m, inf, c, lam, eps_tol, 1e5, h);
        otherwise
            error('limiter_solve: unknown solver %s.', name)
    end
end
