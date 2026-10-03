function red = poisson3d_reduced(data, beta)
%POISSON3D_REDUCED  Null-space-reduced u-only QP of the 3-D Poisson control
%   problem, for Algorithm 2 (di_qp).  Eliminating K*y = M*u by
%   y = K^{-1}*M*u leaves
%
%       min_u (1/2)*u'*Q_red*u + c_red'*u   s.t.  u_a <= u <= u_b,
%
%   with Q_red*u = M*(K^{-1}*(M*(K^{-1}*(M*u)))) + beta*M*u and
%   c_red = -M*(K^{-1}*(M*yd)).  K is factored once by a sparse Cholesky
%   factorization under a nested-dissection ordering; each Q_red product
%   costs four products with M and two solves with K.
%
%   NORMALIZATION.  The returned data are divided
%   by s = ||c_red||_inf, so that ||red.c||_inf = 1.  The minimizer and the
%   active set are unchanged; the stop ||d|| < tol of di_qp, which is
%   absolute, then reads relative to the data.  Without this, the factor
%   beta*h^3 in c_red and Q_red (about 1e-6 and 1e-9 at n = 15) made the
%   stop trigger at a relative gradient of 1e-2 to 1e-1.  The physical
%   reduced objective is s times the returned one, and the physical
%   multipliers are s times those of di_qp.
%
%   INPUTS
%       data   -   struct from poisson3d_data
%       beta   -   control regularization parameter
%
%   OUTPUT
%       red.Q     -   handle u -> Q_red*u / s
%       red.c     -   c_red / s (N x 1)
%       red.s     -   the scale ||c_red||_inf
%       red.G     -   handle u -> y = K^{-1}*M*u (the state)
%       red.adj   -   handle y -> equality multiplier mu = K^{-1}*M*(yd - y),
%                     from the adjoint equation M*(y - yd) + K*mu = 0

    K = data.K;
    M = data.M;

    % Factor K once: K(perm, perm) = L * L' (nested dissection ordering).
    perm = dissect(K);
    [L, chol_flag] = chol(K(perm,perm), 'lower');
    if chol_flag ~= 0, error('poisson3d_reduced: chol(K) failed under dissect ordering.'); end

    Ksolve  = @(v) k_chol_solve(L, perm, v);
    c_phys  = -M*(Ksolve(M*data.yd));
    s       = norm(c_phys, inf);
    red.Q   = @(u) (M*(Ksolve(M*(Ksolve(M*u)))) + beta*(M*u)) / s;
    red.c   = c_phys / s;
    red.s   = s;
    red.G   = @(u) Ksolve(M*u);
    red.adj = @(y) Ksolve(M*(data.yd - y));
end

function x = k_chol_solve(L, perm, r)
%K_CHOL_SOLVE  Apply K^{-1} via the stored sparse Cholesky factor.
%   K(perm, perm) = L * L'  (lower triangular, nested-dissection ordering).
    x       = zeros(length(r), 1);
    x(perm) = L' \ (L \ r(perm));
end
