function [q, d] = nnls_box(M, rhs, tol, q_warm)
%   NNLS_BOX            Closed-form NNLS for a signed-unit active block.
%
%   Solves the nonnegative least-squares subproblem
%
%       min_{q >= 0}  (1/2)*|| M*q - rhs ||^2
%
%   in CLOSED FORM under the precondition that every column of M is a signed
%   unit vector, M(:,j) = +/- e_{i(j)}.  This is exactly the structure of the
%   active inequality block A(eqset,:)' for a BOX-constrained QP A = [I; -I]:
%   each active row is +/- e_i, so the columns of M are signed unit vectors and
%   the NNLS decouples coordinatewise.  With M'*M diagonal the unconstrained
%   minimizer is M'*rhs, and the nonnegativity constraint reduces to a plain
%   clamp, so
%
%       q = max(0, M'*rhs),     d = M*q - rhs.
%
%   The reconstruction is exact even when a coordinate is active at both bounds
%   (columns +e_i and -e_i): the opposite signs recover the residual along that
%   coordinate.  No iteration, no inner tolerance, no scale dependence.
%
%   This is a drop-in replacement for the iterative NNLS solvers (e.g.
%   apgd_lsqnonneg) in di_qp, matching the [q,d] = solver(M, rhs, tol, q_warm)
%   calling convention.  The tol and q_warm arguments are accepted for
%   interface compatibility and ignored (the solve is direct).
%
%   PRECONDITION (NOT checked here; verify once at the call site): each column
%   of M is +/- e_j.  Passing a general M silently returns a wrong q.  Intended
%   only for box-constrained di_qp, where A = [I; -I] guarantees the structure
%   for every active subset, so the precondition is an invariant of A and need
%   not be re-checked per iteration.  This solver does NOT apply to di_qp_eq,
%   where the active block is the PROJECTED P_E*A_eq' and is not signed-unit.
%
% -------------------------------------------------------------------------
%   INPUTS
%       M       -   (n x neff) active block; each column a signed unit vector.
%       rhs     -   n-dimensional right-hand side (the normalized -grad).
%       tol     -   (Ignored) Accepted for interface compatibility.
%       q_warm  -   (Ignored) Accepted for interface compatibility.
%
%   OUTPUTS
%       q       -   neff-dimensional NNLS solution, q >= 0.
%       d       -   n-dimensional residual M*q - rhs.
% -------------------------------------------------------------------------

q = max(0, M' * rhs);
d = M * q - rhs;

end
