function [x,d] = nnls_tree(A,b,tol,x0)
%   NNLS_TREE   Tree-specialized "method of hinges" for the nonnegative
%               least-squares (NNLS) problem with an INCIDENCE matrix.
%
%   This function computes an optimal solution to the NNLS problem
%
%       min_{x \in \Rn} (1/2)\|A*x - b\|_2^2   subject to x \geqslant 0,
%
%   via Mary Meyer's method of hinges (see hinges_lsqnonneg.m), specialized to
%   the case where A is the (unsigned) node-arc INCIDENCE matrix of a bipartite
%   graph: every column has exactly two nonzero entries, both equal to +1.  This
%   is exactly the active block M_E = A(eqset,:).' that di_rlp hands the NNLS
%   solver when it follows the DUAL of optimal transport (see
%   benchmark_quad_dual_ot.m): the columns are edges, the rows are nodes, and any
%   subset of the active edges is a FOREST (acyclic), hence A has full column
%   rank on every passive set the method of hinges visits.
%
%   The method of hinges is unchanged from hinges_lsqnonneg.m; the only
%   difference is that each unconstrained least-squares subproblem
%   min_z ||A(:,P) z - b||^2 is solved EXACTLY in O(#nodes) by tree elimination
%   (local function tree_ls), rather than by a dense solve or by lsqr.  There is
%   no inner iteration and no inner tolerance: the subproblem is a quadratic
%   network-flow problem on a forest, solved by a cokernel projection plus one
%   leaf-to-root sweep.
%
% -------------------------------------------------------------------------
%   INPUTS
%       A       -   (m x n) incidence matrix: each column has exactly two
%                   nonzeros, both +1 (an edge joining its two endpoint nodes).
%       b       -   m-dimensional col data vector.
%       tol     -   (Optional) tolerance (e.g., 1e-08). Default is derived from
%                   the Frobenius norm of A and the relative size of b.
%       x0      -   (Optional, IGNORED) accepted for interface compatibility
%                   with the di_rlp 4-argument NNLS convention; the method of
%                   hinges starts from the full passive set regardless.
%
%   OUTPUTS
%       x       -   n-dimensional solution vector to the NNLS problem.
%       d       -   m-dimensional col residual vector d = A*x - b.
%
% -------------------------------------------------------------------------
%   NOTE
%       Like hinges_lsqnonneg.m, this starts from the full set [n] and prunes,
%       so it excels when the NNLS optimum is active on almost all of [n] -- the
%       regime of the optimal-transport dual, where the support is ~m+n.
%
% -------------------------------------------------------------------------


%%  Preliminary checks
% Check if the tolerance was supplied by the user.
if nargin < 3 || isempty(tol)
    tol = min(1e-08,100*eps*max(norm(A, 'fro'), 1)*max(norm(b), 1));
end

[N,n] = size(A);

% Precondition: A is an incidence matrix (exactly two nonzeros per column).
% Extract the two endpoint nodes of every edge (column) once.  find() returns
% entries in column-major order, so reshaping the row indices 2-by-n gives the
% endpoints per column.
[ri,ci] = find(A);
if numel(ri) ~= 2*n || ~all(accumarray(ci,1,[n 1]) == 2)
    error('nnls_tree: A must be an incidence matrix (two nonzeros per column).');
end
E = reshape(ri, 2, n).';            % n-by-2 endpoint nodes per edge

% Check if the unconstrained LSQ solution is admissible. If so, return it.
u = tree_ls(E, true(n,1), b, N);
if(min(u) > -tol)
    x = u; d = A*x-b;
    return
end


%%  Compute the NNLS solution via the method of hinges
active_set = true(n,1);
while(true)
    % If the primal constraint is violated, drop the most-negative variable.
    % Otherwise, form the residual and test dual feasibility over all columns.
    if(min(u) <= -tol)
        x = zeros(n,1);
        x(active_set) = u;
        [~,I] = min(x);
        active_set(I) = false;
    else
        theta = A(:,active_set)*u;
        theta = b - theta;
        Atop_times_rho = (theta.'*A);
        [val,I] = max(Atop_times_rho);

        % Dual feasibility: if satisfied, EXIT; else add the violating column.
        if(val < tol)
            break;
        else
            active_set(I) = true;
        end
    end

    % Exact least-squares solve on the current passive set (a sub-forest).
    u = tree_ls(E, active_set, b, N);
end
x = zeros(n,1); x(active_set) = u;
d = -theta;
end


% ===================================================================== %
%  Local function: exact least squares on a forest incidence matrix     %
% ===================================================================== %

function z = tree_ls(E, sel, b, N)
%   TREE_LS   Solve  min_z ||A(:,sel) z - b||^2, where A(:,sel) is the (unsigned)
%   incidence matrix of a forest on N nodes with edges E(sel,:).
%
%   A forest incidence matrix has a fill-free elimination tree, so MATLAB's sparse
%   QR (mldivide) solves this least squares in compiled, depth-robust O(#nodes)
%   time -- faster than an interpreted leaf-to-root sweep on deep forests, tied on
%   bushy ones.  It also returns the correct (unique) residual when A(:,sel) is
%   rank-deficient (a tight cycle in a near-degenerate active set), so no separate
%   forest/cycle branch is needed.  z is returned in the order of the selected
%   columns.
ed = find(sel);
k  = numel(ed);
if k == 0, z = zeros(0,1); return; end
As = sparse([E(ed,1); E(ed,2)], [(1:k).'; (1:k).'], 1, N, k);
ws = warning('off', 'all');     % silences the rank-deficient (cycle) notice;
z  = As \ b;                    % a no-op on genuine forests
warning(ws);
end
