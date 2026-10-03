function [S, K, C, F0, fex] = lhtz41_setup(k, n, tau)
%LHTZ41_SETUP  Data of the accuracy test of Liu-Hu-Taitano-Zhang section
%   4.1 on n-by-n cells of [-10,10]^2 with P^k.
%
%   d_t f = Lap f + div(f v), i.e. (17a) with D = I, u = 0 and all other
%   coefficients 1, so b = -v; NIPG with sigma = 1 and penalty h = dx
%   (our choice).  Returns the mesh S (dg_setup), K =
%   decomposition(dx^2*I + tau*A) and C of the scheme (21) (dg_assemble),
%   the initial datum F0 = L2 projection of (32) at t = 1 followed by
%   Zhang-Shu, and fex(t, x, y) = the exact solution (32).
    S  = dg_setup(k, n, 10);
    dx = S.dx;
    Dfun = @(x, y) deal(ones(size(x)), zeros(size(x)), ones(size(x)));
    bfun = @(x, y) deal(-x, -y);
    [A, C] = dg_assemble(S, Dfun, bfun, 1, dx);    % penalty h = dx (ours)
    K  = decomposition(dx^2*speye(S.nb*S.Nc) + tau*A);

    fex = @f32;
    F0  = dg_project(S, @(x, y) f32(1, x, y));
    F0  = dg_limit(F0, S, -inf, []);               % Zhang-Shu only
end


function f = f32(t, x, y)
%F32  Exact solution (32) of LHTZ.
    s = 1 - exp(-2*t);
    f = exp(-(x.^2 + y.^2)/(2*s)) / (2*pi*s);
end
