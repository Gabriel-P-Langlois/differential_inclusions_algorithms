function k = pde_oc_kkt(qp, x, lambda, z)
%PDE_OC_KKT  KKT residuals of one primal-dual point of the QP built by
%   pde_oc_qp, in the convention of
%   https://scaron.info/blog/optimality-conditions-and-numerical-tolerances-in-qp-solvers.html
%   (the convention of qpbenchmark, whose shifted geometric mean the
%   drivers report).  Every solver is scored by this one function, from its
%   own returned (x, lambda, z).
%
%   INPUT
%       qp      -   struct from pde_oc_qp
%       x       -   primal point [y; u]
%       lambda  -   equality multiplier, in the sign convention
%                   P*x + q + A'*lambda + G'*z = 0 at a KKT point
%       z       -   inequality multipliers [z_ub; z_lb], nonnegative
%
%   OUTPUT
%       k.r_p       max(||A*x - b||_inf, ||[G*x - h]_+||_inf)
%       k.r_d       ||P*x + q + A'*lambda + G'*z||_inf
%       k.r_g       |x'*P*x + q'*x + b'*lambda + h'*z|
%       k.r_p_rel   the larger of the two ratios below
%       k.r_d_rel   r_d / max(||P*x||_inf, ||A'*lambda||_inf, ||G'*z||_inf)
%       k.r_g_rel   r_g / max(|x'*P*x|, |q'*x|, |b'*lambda|, |h'*z|)
%       k.eq        ||A*x - b||_inf
%       k.ineq      ||[G*x - h]_+||_inf
%       k.z_min     min(z), the sign check on the inequality multipliers
%
%   The relative forms matter here: the data carry a factor beta*h^d, so the
%   absolute residuals read small for every solver.
%
%   The blog normalizes the primal residual by the terms of the constraints,
%   max(||A*x||_inf, ||b||_inf, ||G*x||_inf, ||h||_inf).  Here b = 0, so that
%   denominator contains ||A*x||_inf, which IS the equality residual, and the
%   ratio would be 1 whatever the point.  Each block is therefore normalized
%   by the terms of its own equation, K*y = B*u and u_a <= u <= u_b:
%
%       ||K*y - B*u||_inf / max(||K*y||_inf, ||B*u||_inf, ||b||_inf)   and
%       ||[G*x - h]_+||_inf / max(||u||_inf, ||h||_inf),
%
%   and r_p_rel is the larger of the two.  A = [K, -B] supplies both blocks.

    if numel(x) ~= size(qp.P, 1) || numel(lambda) ~= size(qp.A, 1) ...
            || numel(z) ~= size(qp.G, 1)
        error('pde_oc_kkt: (x, lambda, z) sizes do not match the QP.')
    end

    x      = full(x(:));
    lambda = full(lambda(:));
    z      = full(z(:));

    Px  = qp.P * x;
    Atl = qp.A' * lambda;
    Gtz = qp.G' * z;

    k.eq   = norm(qp.A * x - qp.b, inf);
    k.ineq = max(0, max(qp.G * x - qp.h));
    k.r_p  = max(k.eq, k.ineq);

    k.r_d = norm(Px + qp.q + Atl + Gtz, inf);

    quad = x' * Px;
    lin  = qp.q' * x;
    beql = qp.b' * lambda;
    hz   = qp.h' * z;
    k.r_g = abs(quad + lin + beql + hz);

    Ky = qp.A(:, 1:qp.N)   * x(1:qp.N);
    Bu = -(qp.A(:, qp.iu)  * x(qp.iu));
    d_eq   = max([norm(Ky, inf), norm(Bu, inf), norm(qp.b, inf), realmin]);
    d_ineq = max([norm(x(qp.iu), inf), norm(qp.h, inf), realmin]);
    k.r_p_rel = max(k.eq / d_eq, k.ineq / d_ineq);

    k.r_d_rel = k.r_d / max([norm(Px, inf), norm(Atl, inf), norm(Gtz, inf), realmin]);
    k.r_g_rel = k.r_g / max([abs(quad), abs(lin), abs(beql), abs(hz), realmin]);

    k.z_min = min(z);
end
