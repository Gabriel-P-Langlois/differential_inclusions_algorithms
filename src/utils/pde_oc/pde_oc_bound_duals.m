function z = pde_oc_bound_duals(rc_u)
%PDE_OC_BOUND_DUALS  Split a net reduced cost of the control block into the
%   nonnegative bound multipliers z = [z_ub; z_lb] of pde_oc_qp.
%
%   With rc_u = (P*x + q + A'*lambda) on the control block, stationarity
%   reads rc_u + z_ub - z_lb = 0, so z_ub = max(-rc_u, 0) and
%   z_lb = max(rc_u, 0).  At most one bound of a control is active, so the
%   split loses nothing.  Solvers that report separate lower and upper
%   multipliers (MOSEK's slx, sux) pass their difference slx - sux, which is
%   the same reduced cost.
%
%   INPUT
%       rc_u    -   reduced cost of the controls (N x 1)
%
%   OUTPUT
%       z       -   [max(-rc_u, 0); max(rc_u, 0)]  (2N x 1, nonnegative)

    rc_u = full(rc_u(:));
    z    = [max(-rc_u, 0); max(rc_u, 0)];
end
