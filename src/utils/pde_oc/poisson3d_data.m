function data = poisson3d_data(n)
%POISSON3D_DATA  Data of the 3-D Poisson control problem shared by all
%   solvers: Q1 stiffness K and consistent Q1 mass M on the uniform
%   n x n x n interior grid of [0,1]^3 (h = 1/(n+1), N = n^3), and the nodal
%   desired state yhat = exp(-64*||x - (0.5,0.5,0.5)||^2) of Pearson &
%   Gondzio (2017), Sec. 5.  Used by results/benchmark_pde_oc_3D.m; the
%   assumptions A1-A3 are listed in its header.
%
%   OUTPUT
%       data.n, data.N, data.h   -   grid size, DOFs, spacing
%       data.K, data.M           -   N x N sparse stiffness and mass
%       data.yd                  -   nodal values of yhat (N x 1)

    h   = 1 / (n + 1);
    N   = n^3;
    e   = ones(n, 1);
    K1D = spdiags([-e, 2*e, -e], [-1,0,1], n, n) / h;
    M1D = spdiags([ e, 4*e,  e], [-1,0,1], n, n) * (h/6);
    K   = kron(kron(K1D,M1D),M1D) + kron(kron(M1D,K1D),M1D) + kron(kron(M1D,M1D),K1D);
    M   = kron(kron(M1D,M1D),M1D);

    x1vec = (1:n)' * h;
    [X1, X2, X3] = ndgrid(x1vec, x1vec, x1vec);
    Yhat = exp(-64 * ((X1 - 0.5).^2 + (X2 - 0.5).^2 + (X3 - 0.5).^2));

    data = struct('n', n, 'N', N, 'h', h, 'K', K, 'M', M, 'yd', Yhat(:));
end
