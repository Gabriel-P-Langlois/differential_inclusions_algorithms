function data = heat2d_data(h, tau, T, A_yhat)
%HEAT2D_DATA  Data of the time-dependent 2-D heat control problem shared by
%   all solvers: Q1 operators on the uniform interior grid of [0,1]^2
%   (n = 1/h - 1 nodes per side), backward Euler over Nt = T/tau steps and
%   trapezoidal weights in time (Pearson & Gondzio 2017, Sec. 3.3), and the
%   desired state yhat(x,t) = A_yhat*t*sin(2*pi*x1*x2) at t_i = i*tau.
%   Used by results/benchmark_pde_oc_2D.m; the assumptions A1-A5 are listed
%   in its header.
%
%   OUTPUT
%       data.n, data.h, data.tau, data.Nt, data.Nsp, data.N
%       data.A_sp, data.M_sp, data.K_sp   -   one backward-Euler step, mass,
%                                             stiffness (Nsp x Nsp)
%       data.Kst, data.Mst, data.M12      -   space-time blocks (N x N)
%       data.K1D, data.M1D                -   1-D Q1 stiffness and mass
%       data.yd                           -   stacked nodal yhat (N x 1)
%       data.fT                           -   right-hand side of the state
%                                             equation (zero: y0 = 0)

    n   = round(1/h) - 1;        % interior grid points per dimension
    Nsp = n^2;                   % spatial DOFs
    Nt  = round(T / tau);        % time steps
    N   = Nsp * Nt;              % space-time DOFs (per variable block)

    % Q1 spatial operators on [0,1]^2 and the space-time blocks.
    e   = ones(n, 1);
    K1D = spdiags([-e, 2*e, -e], [-1,0,1], n, n) / h;
    M1D = spdiags([ e, 4*e,  e], [-1,0,1], n, n) * (h/6);

    K_sp = kron(K1D, M1D) + kron(M1D, K1D);     % Nsp x Nsp, SPD
    M_sp = kron(M1D, M1D);                       % consistent Q1 mass
    A_sp = M_sp + tau * K_sp;                    % backward-Euler step operator

    I_t   = speye(Nt);
    sub_t = spdiags(ones(Nt,1), -1, Nt, Nt);     % first subdiagonal in time
    Kst   = kron(I_t, A_sp) - kron(sub_t, M_sp); % block lower-bidiagonal
    Mst   = kron(I_t, M_sp);

    theta = ones(Nt, 1);  theta(1) = 0.5;  theta(Nt) = 0.5;   % trapezoid weights
    M12   = kron(spdiags(theta, 0, Nt, Nt), M_sp);

    % Desired state yhat(x,t) = A_yhat*t*sin(2*pi*x1*x2), levels t_i = i*tau.
    x1vec    = (1:n)' * h;
    [X1, X2] = ndgrid(x1vec, x1vec);
    g_sp     = reshape(sin(2*pi*X1.*X2), [], 1);
    yd       = A_yhat * kron((1:Nt)' * tau, g_sp);

    data = struct('n', n, 'h', h, 'tau', tau, 'Nt', Nt, 'Nsp', Nsp, 'N', N, ...
                  'A_sp', A_sp, 'M_sp', M_sp, 'K_sp', K_sp, 'Kst', Kst, ...
                  'Mst', Mst, 'M12', M12, 'K1D', K1D, 'M1D', M1D, ...
                  'yd', yd, 'fT', zeros(N, 1));
end
