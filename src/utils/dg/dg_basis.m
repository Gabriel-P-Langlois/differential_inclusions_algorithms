function [P, Ps, Pt, modes] = dg_basis(k, s, t)
%DG_BASIS  Orthonormal Legendre basis of P^k (total degree <= k) on the
%   reference cell [-1/2, 1/2]^2, as in Liu-Hu-Taitano-Zhang section 3.1.
%
%   [P, Ps, Pt, modes] = dg_basis(k, s, t) evaluates the nb = (k+1)(k+2)/2
%   basis functions phi_r(s,t) = L_a(s)*L_b(t), (a,b) = modes(r,:), at the
%   points (s(i), t(i)), with L_n(s) = sqrt(2n+1)*P_n(2s) the Legendre
%   polynomials orthonormal on [-1/2, 1/2].  P, Ps, Pt are numel(s)-by-nb:
%   values, d/ds and d/dt.  Modes are ordered by total degree; mode 1 is the
%   constant 1, so the first coefficient of a cell is its average.

    s = s(:);  t = t(:);
    modes = zeros(0, 2);
    for d = 0:k
        for a = d:-1:0
            modes(end+1, :) = [a, d - a]; %#ok<AGROW>
        end
    end
    [Ls, dLs] = leg1d(k, s);
    [Lt, dLt] = leg1d(k, t);
    ia = modes(:,1) + 1;  ib = modes(:,2) + 1;
    P  = Ls(:, ia)  .* Lt(:, ib);
    Ps = dLs(:, ia) .* Lt(:, ib);
    Pt = Ls(:, ia)  .* dLt(:, ib);
end


function [L, dL] = leg1d(k, s)
%LEG1D  Orthonormal Legendre polynomials L_0..L_k on [-1/2, 1/2] and their
%   derivatives, by the three-term recurrence and P'_{n+1} = P'_{n-1} +
%   (2n+1) P_n (exact at the end points).
    x  = 2*s;
    np = numel(x);
    Pn  = zeros(np, k+1);
    dPn = zeros(np, k+1);
    Pn(:,1) = 1;
    if k >= 1
        Pn(:,2) = x;  dPn(:,2) = 1;
    end
    for n = 1:k-1
        Pn(:,n+2)  = ((2*n+1)*x.*Pn(:,n+1) - n*Pn(:,n)) / (n+1);
        dPn(:,n+2) = dPn(:,n) + (2*n+1)*Pn(:,n+1);
    end
    c  = sqrt(2*(0:k) + 1);
    L  = Pn .* c;
    dL = 2 * dPn .* c;
end
