function x = limiter_exact_w(u, m, M, w)
%LIMITER_EXACT_W  Reference minimizer of the weighted limiter QP
%
%       min (1/2)*||x - u||_W^2  s.t.  w'*x = w'*u,  m <= x <= M,
%
%   W = diag(w), w > 0, by bisection on the weighted root equation
%   sum(w.*clip(u + nu)) = w'*u.  Independent of the solvers, so it serves
%   as their oracle.
%
%   M may be +Inf.  The upper bracket M - min(u) is then infinite, and 0 is
%   used instead: g(0) = w'*max(u, m) - w'*u >= 0.
    b = w'*u;
    g = @(nu) w'*min(max(u + nu, m), M) - b;
    lo = m - max(u);
    if isinf(M)
        hi = 0;
    else
        hi = M - min(u);
    end
    for k = 1:200
        mid = 0.5*(lo + hi);
        if g(mid) < 0, lo = mid; else, hi = mid; end
    end
    x = min(max(u + 0.5*(lo + hi), m), M);
end
