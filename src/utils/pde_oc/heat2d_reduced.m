function red = heat2d_reduced(data, beta)
%HEAT2D_REDUCED  Null-space-reduced u-only QP of the time-dependent 2-D heat
%   control problem, for Algorithm 2 (di_qp).  Eliminating the state
%   y = Kst^{-1}(tau*Mst*u + fT) leaves
%
%       min_u (1/2)*u'*Q_red*u + c_red'*u   s.t.  u_a <= u <= u_b,
%
%   with G = Kst^{-1}*tau*Mst, Q_red = tau*G'*M12*G + beta*tau*M12 and
%   c_red = tau*G'*M12*(y_f - yd), y_f = Kst^{-1}*fT.  On the uniform grid
%   K1D and M1D share the orthogonal DST-I eigenvectors S, so M_sp, K_sp and
%   A_sp are diagonal in the 2-D sine basis; Kst^{-1} and Kst^{-T} are one
%   scalar recurrence per mode, and each Q_red product costs two 2-D sine
%   transforms of the stacked array.  Exact on this grid only.
%
%   NORMALIZATION.  The returned data are divided
%   by s = ||c_red||_inf, so that ||red.c||_inf = 1.  The minimizer and the
%   active set are unchanged; the stop ||d|| < tol of di_qp, which is
%   absolute, then reads relative to the data.  Without this, the factor
%   beta*tau*h^2 in c_red and Q_red (about 1e-6 at h = 2^-4) made the stop
%   trigger at a relative gradient of 1e-2 to 1e-1.  The physical reduced
%   objective is s times the returned one, and the physical multipliers
%   are s times those of di_qp.
%
%   INPUTS
%       data   -   struct from heat2d_data
%       beta   -   control regularization parameter
%
%   OUTPUT
%       red.Q     -   handle u -> Q_red*u / s
%       red.c     -   c_red / s (N x 1)
%       red.s     -   the scale ||c_red||_inf
%       red.G     -   handle u -> G*u (the state is y = G*u + red.y_f)
%       red.y_f   -   Kst^{-1}*fT (zero here)
%       red.adj   -   handle y -> equality adjoint
%                     lambda = -Kst^{-T}*tau*M12*(y - yd)

    n   = data.n;
    Nt  = data.Nt;
    tau = data.tau;
    yd  = data.yd;

    [S, mu, den] = sine_modes(data.K1D, data.M1D, tau);
    theta = ones(1, Nt);  theta([1, Nt]) = 0.5;          % trapezoid weights
    to_modal = @(v)  sine2(v, S, n, Nt);                 % Nsp x Nt modal array
    to_phys  = @(Vh) reshape(sine2(Vh, S, n, Nt), [], 1);
    Gm   = @(Uh) kst_solve_modal(tau * mu .* Uh, mu, den);   % G   in modal form
    Gtm  = @(Wh) tau * mu .* kst_adj_modal(Wh, mu, den);     % G'  in modal form
    M12m = @(Vh) (mu * theta) .* Vh;                         % M12 in modal form

    y_f    = to_phys(kst_solve_modal(to_modal(data.fT), mu, den));
    c_phys = to_phys(tau * Gtm(M12m(to_modal(y_f - yd))));
    s      = norm(c_phys, inf);
    Qhat   = @(Uh) (tau * Gtm(M12m(Gm(Uh))) + beta*tau*M12m(Uh)) / s;

    red.Q   = @(u) to_phys(Qhat(to_modal(u)));
    red.c   = c_phys / s;
    red.s   = s;
    red.G   = @(u) to_phys(Gm(to_modal(u)));
    red.y_f = y_f;
    red.adj = @(y) -to_phys(kst_adj_modal(tau * M12m(to_modal(y - yd)), mu, den));
end

function [S, mu, den] = sine_modes(K1D, M1D, tau)
%SINE_MODES  Orthogonal DST-I matrix S (S = S' = S^{-1}) and the modal
%   eigenvalues of the 2D operators on the uniform Dirichlet grid:
%   S*K1D*S = diag(k), S*M1D*S = diag(m), so in the 2D sine basis
%   M_sp = diag(mu) with mu = kron(m,m), K_sp = diag(kron(k,m) + kron(m,k)),
%   and A_sp = M_sp + tau*K_sp = diag(den).
    n = size(K1D, 1);
    j = (1:n)';
    S = sqrt(2/(n+1)) * sin(j*j' * pi/(n+1));
    Dk = S*K1D*S;  Dm = S*M1D*S;
    k = diag(Dk);  m = diag(Dm);
    assert(norm(Dk - diag(k), 'fro') <= 1e-12*norm(k) && ...
           norm(Dm - diag(m), 'fro') <= 1e-12*norm(m), ...
           'sine_modes: DST-I does not diagonalize K1D and M1D.')
    mu  = kron(m, m);
    den = mu + tau*(kron(k, m) + kron(m, k));
end

function V = sine2(v, S, n, Nt)
%SINE2  2D sine transform S*X*S of each time level X (n x n) of the stacked
%   vector or array v (n^2*Nt entries, x1 fastest); returns an n^2 x Nt
%   array.  S is orthogonal and symmetric, so SINE2 is its own inverse.
    X = reshape(v, n, n, Nt);
    V = reshape(pagemtimes(S, pagemtimes(X, S)), n*n, Nt);
end

function Yh = kst_solve_modal(Rh, mu, den)
%KST_SOLVE_MODAL  Solve Kst*y = r in the sine basis: den.*y_i - mu.*y_{i-1}
%   = r_i, y_0 = 0, one scalar recurrence per mode (columns = time levels).
    Nt = size(Rh, 2);
    Yh = zeros(size(Rh));
    Yh(:,1) = Rh(:,1) ./ den;
    for i = 2:Nt
        Yh(:,i) = (mu .* Yh(:,i-1) + Rh(:,i)) ./ den;
    end
end

function Zh = kst_adj_modal(Wh, mu, den)
%KST_ADJ_MODAL  Solve Kst'*z = w in the sine basis: den.*z_i - mu.*z_{i+1}
%   = w_i, z_{Nt+1} = 0, one scalar recurrence per mode, backward in time.
    Nt = size(Wh, 2);
    Zh = zeros(size(Wh));
    Zh(:,Nt) = Wh(:,Nt) ./ den;
    for i = Nt-1:-1:1
        Zh(:,i) = (mu .* Zh(:,i+1) + Wh(:,i)) ./ den;
    end
end
