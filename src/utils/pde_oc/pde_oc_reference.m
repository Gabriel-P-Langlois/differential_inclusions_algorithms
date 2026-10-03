function ref = pde_oc_reference(Qfun, c, u_di, p_di, u_a, u_b, tol_act, tol_cg, maxit_cg, tol_sign)
%PDE_OC_REFERENCE  Reference solution of the box-constrained reduced QP,
%   built from the final active set of di_qp and checked a posteriori.
%
%   The reduced problem is
%
%       min_u (1/2)*u'*Q*u + c'*u   s.t.  u_a <= u <= u_b,
%
%   with Q given as a handle v -> Q*v.  di_qp returns u_di and the dual
%   p_di of the rows of A = [I; -I], b = [u_b; -u_a]: p_di(i) is the
%   multiplier of u_i <= u_b and p_di(N+i) that of -u_i <= -u_a.  A
%   coordinate is taken as active at its upper bound if its multiplier is
%   positive or u_di lies within tol_act of u_b, and likewise at the lower
%   bound.  The active coordinates are fixed at their bounds and the free
%   block is solved by conjugate gradients (pcg) on
%
%       Q_FF * u_F = -(c + Q*u_A)_F,
%
%   warm started at u_di(F), to relative residual tol_cg and at most
%   maxit_cg iterations.  Each pcg iteration costs one product with Q.
%
%   The result is accepted as a reference only if the KKT conditions hold
%   a posteriori: every free coordinate lies strictly inside (u_a, u_b),
%   and the gradient g = Q*u_ref + c satisfies g >= -tol_sign*||g||_inf on
%   the lower-bound set and g <= tol_sign*||g||_inf on the upper-bound set.
%   The check does not depend on how the active set was found.
%
% -------------------------------------------------------------------------
%   INPUTS
%       Qfun        handle v -> Q*v
%       c           reduced linear term (N x 1)
%       u_di, p_di  output of di_qp (N x 1 and 2N x 1)
%       u_a, u_b    scalar bounds, u_a < u_b
%       tol_act     membership tolerance for the bounds (the tol of di_qp)
%       tol_cg      pcg relative residual tolerance
%       maxit_cg    pcg iteration cap
%       tol_sign    relative tolerance of the gradient sign check
%
%   OUTPUT  struct ref with fields
%       u            reference control (N x 1)
%       g            gradient Q*u + c at ref.u
%       fval         (1/2)*u'*Q*u + c'*u at ref.u
%       pass         true if the a posteriori KKT check holds
%       at_lb, at_ub, free   logical masks of the active set used
%       n_lb, n_ub, n_free   their sizes
%       slack_min    min over the free coordinates of min(u - u_a, u_b - u)
%                    (Inf if no coordinate is free)
%       sign_viol    largest sign violation of g on the active set (0 if none)
%       cg_flag, cg_relres, cg_iter   pcg outputs (flag 0 = converged)
%       relres_true  ||g(F)||_2 / ||rhs||_2 from a fresh product with Q
%
% -------------------------------------------------------------------------

    N = numel(u_di);
    if numel(p_di) ~= 2*N
        error('pde_oc_reference: p_di must have 2*numel(u_di) entries.')
    end

    at_ub = (p_di(1:N) > 0) | (u_di >= u_b - tol_act);
    at_lb = (p_di(N+1:2*N) > 0) | (u_di <= u_a + tol_act);
    if any(at_ub & at_lb)
        error('pde_oc_reference: a coordinate is active at both bounds.')
    end
    free = ~(at_ub | at_lb);

    u_A = zeros(N, 1);
    u_A(at_ub) = u_b;
    u_A(at_lb) = u_a;

    u_ref = u_A;
    if any(free)
        w   = Qfun(u_A);
        rhs = -(c(free) + w(free));
        Qff = @(v) restrict(Qfun(embed(v, free, N)), free);
        [u_F, cg_flag, cg_relres, cg_iter] = pcg(Qff, rhs, tol_cg, maxit_cg, [], [], u_di(free));
        u_ref(free) = u_F;
    else
        rhs = [];
        cg_flag = 0;  cg_relres = 0;  cg_iter = 0;
    end

    Qu   = Qfun(u_ref);
    g    = Qu + c;
    fval = 0.5 * (u_ref' * Qu) + c' * u_ref;

    if any(free)
        slack_min   = min([u_ref(free) - u_a; u_b - u_ref(free)]);
        relres_true = norm(g(free)) / max(norm(rhs), realmin);
    else
        slack_min   = Inf;
        relres_true = 0;
    end
    sign_viol = max([0; -g(at_lb); g(at_ub)]);
    pass = (slack_min > 0) && (sign_viol <= tol_sign * norm(g, inf));

    ref = struct('u', u_ref, 'g', g, 'fval', fval, 'pass', pass, ...
                 'at_lb', at_lb, 'at_ub', at_ub, 'free', free, ...
                 'n_lb', nnz(at_lb), 'n_ub', nnz(at_ub), 'n_free', nnz(free), ...
                 'slack_min', slack_min, 'sign_viol', sign_viol, ...
                 'cg_flag', cg_flag, 'cg_relres', cg_relres, 'cg_iter', cg_iter, ...
                 'relres_true', relres_true);
end

function z = embed(v, free, N)
    z = zeros(N, 1);
    z(free) = v;
end

function v = restrict(z, free)
    v = z(free);
end
