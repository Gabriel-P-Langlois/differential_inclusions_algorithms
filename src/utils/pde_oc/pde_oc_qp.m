function qp = pde_oc_qp(H, fvec, Aeq, beq, u_a, u_b, N)
%PDE_OC_QP  Pack the full (y, u) QP of the PDE-OC benchmarks in the form
%
%       min_x  (1/2)*x'*P*x + q'*x   s.t.   G*x <= h,   A*x = b,
%
%   with x = [y; u], only the control block bounded.  The inequality rows
%   are stacked upper bounds first, so a dual vector z = [z_ub; z_lb] has
%   G'*z = [0; z_ub - z_lb].  One place owns this ordering; the residuals
%   of pde_oc_kkt and the sign split of pde_oc_bound_duals follow it.
%
%   INPUT
%       H, fvec, Aeq, beq   -   full QP data as the drivers assemble them
%       u_a, u_b            -   scalar control bounds
%       N                   -   DOFs per variable block (y and u)
%
%   OUTPUT
%       qp.P, qp.q, qp.A, qp.b, qp.G, qp.h   -   the arrays above
%       qp.N                                 -   N
%       qp.iu                                -   index range of the controls

    qp.P  = H;
    qp.q  = full(fvec);
    qp.A  = Aeq;
    qp.b  = full(beq);
    qp.G  = [sparse(N, N),  speye(N); ...
             sparse(N, N), -speye(N)];
    qp.h  = [u_b * ones(N, 1); -u_a * ones(N, 1)];
    qp.N  = N;
    qp.iu = (N+1 : 2*N)';
end
