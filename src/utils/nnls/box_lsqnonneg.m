function [q,d] = box_lsqnonneg(A,b,~,~)
%   BOX_LSQNONNEG   Closed-form nonnegative least squares when A is a signed
%                   column selection.
%
%   Solves
%
%   (1) min_{q >= 0}  (1/2)*||A*q - b||^2
%
%   when every column of A has exactly one nonzero entry, equal to +1 or -1,
%   and no two columns share a row, so that A'*A = I.  Then (1) separates
%   into one scalar problem per column and q = max(0, A'*b).  This is the
%   NNLS subproblem of di_qp on box constraints: the active rows of
%   [I; -I] are signed unit vectors, and at most one bound of each
%   coordinate is active.  Any other A raises an error.
%
%   The calling convention is that of the other solvers in this folder,
%   [q, d] = solver(A, b, tol, q_warm), with d = A*q - b; the tolerance and
%   the warm start are not used.
%
% -------------------------------------------------------------------------

    k = size(A, 2);
    if nnz(A) ~= k || any(full(sum(abs(A), 1)) ~= 1) || any(full(sum(abs(A), 2)) > 1)
        error('box_lsqnonneg: A must be a signed column selection (A''*A = I).')
    end
    q = max(0, A.' * b);
    d = A * q - b;
end
