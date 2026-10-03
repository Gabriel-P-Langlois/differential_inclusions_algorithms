function [x, iters] = dr_lhtz(u, m, M, c, lam, eps_tol, maxit, h)
%DR_LHTZ  Algorithm DR of Liu-Hu-Taitano-Zhang (CAMWA 192, 2025) for the
%   limiter QP (24),
%
%       min (1/2)*||x - u||^2  s.t.  sum(x) = sum(u),  m <= x <= M:
%
%   the relaxed Douglas-Rachford iteration (30) from y0 = u, x0 = S(u), with
%   S the cut-off (29), stopped by step 4, h*||y^{k+1} - y^k||_2 < eps_tol
%   (the norm ||.||_2h of Remark 3 with d = 2).  c and lam come from
%   dr_params.  M may be +Inf.
    N = numel(u);
    b = sum(u);
    y = u;
    for iters = 1:maxit
        x = min(max(y, m), M);
        z = 2*x - y;
        q = c*z + (1-c)*u;
        ynew = lam*(q - (sum(q) - b)/N) + y - lam*x;
        delta = norm(ynew - y);
        y = ynew;
        if h*delta < eps_tol, break, end
    end
    if h*delta >= eps_tol
        warning('dr_lhtz: no convergence within maxit = %d iterations.', maxit)
    end
    x = min(max(y, m), M);
end
