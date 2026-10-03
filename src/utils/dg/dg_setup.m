function S = dg_setup(k, n, L)
%DG_SETUP  Uniform n-by-n DG mesh of [-L, L]^2 with P^k reference data,
%   following Liu-Hu-Taitano-Zhang section 3.1: square cells, Legendre
%   orthonormal modal basis on [-1/2,1/2]^2, tensor (k+1)-point Gauss
%   quadrature.
%
%   Cells are numbered c = i + (j-1)*n, i the x-index and j the y-index.
%   Coefficients are stored nb-by-Nc; column c holds cell c, row 1 its
%   average.  With the orthonormal basis the mass matrix is dx^2 * I.
%
%   Fields:
%     k, n, L, dx, Nc, nb, modes
%     g, w1          1D Gauss points and weights on [-1/2,1/2] (sum 1)
%     wq             tensor weights (nq-by-1, sum 1)
%     Phi, dPs, dPt  basis values and reference gradients at the volume
%                    points (nq-by-nb)
%     tr.E/W/N/S     traces on the east/west/north/south reference edges:
%                    fields P, dS, dT (np-by-nb), np = k+1
%     Xq, Yq         physical volume points (nq-by-Nc)
%     vm, vp         interior vertical faces: left and right cells
%     vX, vY         their physical face points (np-by-Nfv)
%     hm, hp         interior horizontal faces: bottom and top cells
%     hX, hY         their physical face points (np-by-Nfh)
%   The face normal points from the cell of lower index (vm, hm) into the
%   cell of higher index (vp, hp): (1,0) vertical, (0,1) horizontal.

    S.k = k;  S.n = n;  S.L = L;
    S.dx = 2*L/n;
    S.Nc = n^2;
    S.nb = (k+1)*(k+2)/2;

    [g, w1] = gauss01(k+1);
    S.g = g;  S.w1 = w1;
    [sq, tq] = ndgrid(g, g);
    S.wq = reshape(w1*w1.', [], 1);
    [S.Phi, S.dPs, S.dPt, S.modes] = dg_basis(k, sq(:), tq(:));

    half = 0.5*ones(k+1, 1);
    [S.tr.E.P, S.tr.E.dS, S.tr.E.dT] = dg_basis(k,  half, g);
    [S.tr.W.P, S.tr.W.dS, S.tr.W.dT] = dg_basis(k, -half, g);
    [S.tr.N.P, S.tr.N.dS, S.tr.N.dT] = dg_basis(k, g,  half);
    [S.tr.S.P, S.tr.S.dS, S.tr.S.dT] = dg_basis(k, g, -half);

    dx = S.dx;
    [ii, jj] = ndgrid(1:n, 1:n);
    xc = -L + (ii(:) - 0.5)*dx;
    yc = -L + (jj(:) - 0.5)*dx;
    S.Xq = xc.' + dx*sq(:);
    S.Yq = yc.' + dx*tq(:);

    [iv, jv] = ndgrid(1:n-1, 1:n);
    S.vm = iv(:) + (jv(:) - 1)*n;
    S.vp = S.vm + 1;
    S.vX = repmat((-L + iv(:)*dx).', k+1, 1);
    S.vY = yc(S.vm).' + dx*g;

    [ih, jh] = ndgrid(1:n, 1:n-1);
    S.hm = ih(:) + (jh(:) - 1)*n;
    S.hp = S.hm + n;
    S.hX = xc(S.hm).' + dx*g;
    S.hY = repmat((-L + jh(:)*dx).', k+1, 1);
end


function [g, w] = gauss01(np)
%GAUSS01  np-point Gauss-Legendre rule on [-1/2, 1/2], weights summing to 1
%   (Golub-Welsch).
    b = (1:np-1) ./ sqrt(4*(1:np-1).^2 - 1);
    [V, D] = eig(diag(b, 1) + diag(b, -1));
    [x, i] = sort(diag(D));
    g = x/2;
    w = (V(1, i).^2).';
end
