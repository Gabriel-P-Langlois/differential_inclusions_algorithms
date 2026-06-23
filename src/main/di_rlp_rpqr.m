function [X,P,fvals,flags,bpts] = di_rlp_rpqr(At,b,c,tvec,x0,tol,maxit)
%   DI_RLP_RPQR   Regularization-path solver for L2-regularized linear
%                 programs via differential inclusions and a carried QR.
%
%   Solves, for every t_k in a path TVEC = [t_1,...,t_K],
%
%       min_x  (t_k/2)||x||^2 + <c,x>   s.t.  A*x <= b,   A = At.',
%
%   returning x*(t_k) as column k of X.  It is the path companion to
%   DI_RLP: instead of re-solving each level from scratch, it runs the
%   breakpoint loop while carrying one dense QR factorization (Q,R) of the
%   active block, updated by single-column inserts and deletes across both
%   breakpoints and levels.  A level warm-starts from the previous x, so
%   the hand-off needs no factor update.
%
%   ORIENTATION.  Only the transpose At is stored; its columns are the
%   constraint normals.  Every access is a column operation or a
%   transpose-multiply, so the sparse At is never row-indexed: the matvecs
%   use At.'*x and At.'*d, the active block is At(:,cols), and a new QR
%   column is At(:,j).
%
%   QR UPDATES.  Q is the full nv-by-nv orthogonal factor and R the
%   nv-by-k tall upper-triangular factor.  Columns enter and leave via the
%   internal Givens routines insertCol/deleteCol.  The passive-set solve is
%   z = linsolve(R, Q.'*rhs, opts) with opts.UT set once; R and Q.'*rhs are
%   passed whole, so no submatrix is copied per solve.
%
%   INNER SOLVE.  The NNLS over the active block is a warm-started
%   active-set method (drop the most-negative variable, add the most
%   dual-infeasible one).  The active block is assumed unit incidence (two
%   nonzeros per column); the rank guard skips dependent columns.
%
%   t = 0 ENDPOINT.  A final zero entry is the unregularized LP.  It uses
%   the same carried-QR loop (rhs -c, stopping when the ray is unblocked),
%   reusing the factor from the smallest t > 0 level.
%
% -------------------------------------------------------------------------
%   INPUTS
%       At    - (nv x mc) transpose of the constraint matrix; column j is
%               the normal of constraint j (constraints read At.'*x <= b).
%               Assumed unit incidence (two nonzeros per column).
%       b     - mc-vector, right-hand side.
%       c     - nv-vector, linear term.
%       tvec  - K-vector of parameters t_k >= 0; only the last may be 0.
%               Decreasing toward 0 is intended (warm-start payoff), but
%               any order is correct since the feasible set is fixed.
%       x0    - (Optional) feasible point At.'*x0 <= b; else di_phase1.
%       tol   - (Optional) tolerance.  Default as in di_rlp.
%       maxit - (Optional) per-level cap.  Default max(1e6,100*nv).
%
%   OUTPUTS
%       X     - (nv x K) solutions; X(:,k) = x*(t_k).
%       P     - (mc x K) KKT multipliers; P(:,k) = p*(t_k).
%       fvals - (1 x K) objective values (t_k/2)||x||^2 + c'x.
%       flags - (1 x K) per level: 1 success, -1 maxit, 0 infeasible
%               (or unbounded at the t = 0 LP level).
%       bpts  - (1 x K) breakpoint count per level.
%
% -------------------------------------------------------------------------


%% Preliminary checks
if nargin < 4
    error('di_rlp_rpqr: Not enough input arguments.');
end
if ~iscolumn(b) || ~iscolumn(c)
    error('di_rlp_rpqr: b and c must be column vectors.');
end
[nv,mc] = size(At);
if size(b,1) ~= mc
    error('di_rlp_rpqr: size(At,2) must match length(b).');
end
if size(c,1) ~= nv
    error('di_rlp_rpqr: size(At,1) must match length(c).');
end
if ~isvector(tvec) || any(tvec < 0)
    error('di_rlp_rpqr: tvec must be a nonnegative vector.');
end
if numel(tvec) > 1 && any(tvec(1:end-1) <= 0)
    error('di_rlp_rpqr: only the LAST entry of tvec may be zero.');
end
tvec = tvec(:).';
K = numel(tvec);

if nargin < 6 || isempty(tol)
    tol = min(1e-08, 100*eps*max(norm(At,'fro'),1)*max(norm(b),1));
end
if nargin < 7 || isempty(maxit)
    maxit = max(1e6, 100*nv);
end
maxit_lp = 200*nv;                       % cap for the t = 0 LP level

% Feasible start.  di_phase1 wants A = At.' and is reached only when no x0
% is supplied (then uses di_phase1's default NNLS solver).
if nargin < 5 || isempty(x0)
    [x, flag_feas] = di_phase1(At.', b, tol, maxit);
    if (flag_feas == 0) || (flag_feas == -1)
        fprintf('di_rlp_rpqr: the program is infeasible within tolerance.\n');
        X = []; P = []; fvals = []; flags = 0; bpts = [];
        return;
    end
else
    x = x0;
end


%% Outputs and carried state
X     = zeros(nv, K);
P     = zeros(mc, K);
fvals = zeros(1, K);
flags = zeros(1, K);
bpts  = zeros(1, K);

residual = b - At.'*x;                    % maintained incrementally below
if any(-residual > tol)
    error('di_rlp_rpqr: the initial point is infeasible within tolerance.');
end
eqset = abs(residual) < tol;             % active inequality rows

% Carried factorization of the active block B = At(:,cols):
%   cols : global indices of the strictly-positive (passive) columns.
%   qc   : the matching nonnegative multipliers.
%   Q,R  : dense factors; Q is nv-by-nv (kept square for insertCol/
%          deleteCol) and R is nv-by-k tall upper-triangular.
cols = zeros(1,0);
qc   = zeros(0,1);
Q    = eye(nv);
R    = zeros(nv,0);

opts.UT = true;                          % R is upper triangular

reconcile();                             % drop any carried col not in eqset


%% Path loop
for k = 1:K
    t = tvec(k);

    % --- t = 0: unregularized LP endpoint ---
    if t == 0
        reconcile();
        cnt = 0;  flg = 1;
        while true
            cnt = cnt + 1;

            % Descent direction and maximum admissible step.
            d = solve_nnls(-c);
            A_times_d = At.'*d;
            ratio = residual ./ A_times_d;
            ratio(eqset | (A_times_d <= 0)) = inf;
            timestep = min(ratio);

            % Ray unblocked: optimum (d = 0) or unbounded (d ~= 0).
            if timestep*tol > 1
                if norm(d, inf) >= tol
                    flg = 0;             % unbounded
                end
                break;
            end

            % Otherwise step to the breakpoint and continue.
            x        = x + timestep*d;
            residual = residual - timestep*A_times_d;
            eqset    = abs(residual) < tol;
            reconcile();

            if cnt == maxit_lp
                flg = -1;
                break;
            end
        end

        X(:,k) = x;
        if flg == 1
            p = zeros(mc,1);  p(cols) = qc;  P(:,k) = p;
        end                              % else P(:,k) stays zero
        fvals(k) = c'*x;
        flags(k) = flg;
        bpts(k)  = cnt;
        continue;
    end

    % --- t > 0: differential-inclusions breakpoint loop ---
    reconcile();
    cnt = 0;  flg = 1;
    while true
        cnt = cnt + 1;

        % Descent direction and maximum admissible step.
        d = solve_nnls(-(t*x + c));
        A_times_d = At.'*d;
        ratio = residual ./ A_times_d;
        ratio(eqset | (A_times_d <= 0)) = inf;
        timestep = min(ratio);

        % Unconstrained minimizer along d sits at step 1/t.
        if t*timestep >= 1
            x        = x + d/t;
            residual = residual - (1/t)*A_times_d;
            eqset    = abs(residual) < tol;
            reconcile();
            solve_nnls(-(t*x + c));      % refresh multipliers at optimum
            break;
        end

        % Otherwise step to the breakpoint and continue.
        x        = x + timestep*d;
        residual = residual - timestep*A_times_d;
        eqset    = abs(residual) < tol;
        reconcile();

        if cnt == maxit
            flg = -1;
            break;
        end
    end

    X(:,k)   = x;
    p        = zeros(mc,1);  p(cols) = qc;  P(:,k) = p;
    fvals(k) = 0.5*t*(x'*x) + c'*x;
    flags(k) = flg;
    bpts(k)  = cnt;
end


% ===================================================================== %
%  Nested functions (share cols, qc, Q, R, eqset, At, tol, opts)        %
% ===================================================================== %

    function reconcile()
        %   Drop carried passive columns that left the active set, keeping
        %   (cols, qc, Q, R) consistent.  Delete high-to-low.
        if isempty(cols), return; end
        keep = eqset(cols);
        if all(keep), return; end
        for pdrop = sort(find(~keep), 'descend')
            del_col(pdrop);
        end
    end

    function d = solve_nnls(rhs)
        %   Warm-started active-set NNLS over the QR-carried passive set,
        %   min_{q>=0} ||At(:,eqset)*q - rhs||.  Pivoting follows the
        %   method of hinges: drop the most-negative variable, add the
        %   most dual-infeasible one.  Returns the descent d = rhs - B*qc.
        %
        %   eqset is fixed here, so the active normals Aeq and indices
        %   eqidx are gathered once; the inner loop then needs no find
        %   over all mc constraints and no row gather.
        eqidx = find(eqset);
        Aeq   = At(:, eqidx);            % active normals as columns
        blk   = false(numel(eqidx), 1);  % dependent candidates (local)
        resid = rhs;
        while true
            % Least squares on the passive set; drop a negative.
            if isempty(cols)
                z = zeros(0,1);
            else
                tmp = Q.' * rhs;
                % R is tall upper-trapezoidal (zeros below row k), so
                % linsolve uses the leading k-by-k block; passing R and
                % tmp whole avoids copying R(1:k,1:k).
                z = linsolve(R, tmp, opts);
            end
            if ~isempty(z) && min(z) <= -tol
                [~, pdrop] = min(z);
                del_col(pdrop);
                continue;
            end
            qc = z;                      % admissible LS solution

            % Residual via column access: resid = rhs - At(:,cols)*qc.
            if isempty(cols)
                resid = rhs;
            else
                resid = rhs - At(:,cols)*qc;
            end

            % Dual feasibility: correlate the active normals with resid.
            % cols give 0 (resid _|_ range B), so only blocked candidates
            % are excluded; jadd is read straight off eqidx.
            if isempty(eqidx), break; end
            w = Aeq.' * resid;
            w(blk) = -inf;
            [wmax, loc] = max(w);
            if wmax < tol, break; end

            jadd = eqidx(loc);
            if ~add_col(jadd)            % dependent: skip it
                blk(loc) = true;
            end
        end

        if isempty(cols)
            d = rhs;
        else
            d = resid;                   % already formed above
        end
    end

    function ok = add_col(j)
        %   Append normal At(:,j) to B and update (Q,R) by an internal
        %   Givens insert.  Returns false (undoing it) if the column is
        %   dependent on the current ones (new pivot below tol).  At(:,j)
        %   has two nonzeros, so Q.'*At(:,j) is a combination of two rows
        %   of Q, formed in O(nv) rather than a dense O(nv^2) matvec.
        [ii,~,vv] = find(At(:,j));
        loc = numel(cols) + 1;
        R(:,loc) = Q(ii,:).' * vv;
        [Q,R] = matlab.internal.math.insertCol(Q, R, loc);
        if abs(R(loc,loc)) < tol         % dependent: undo the insert
            R(:,loc) = [];
            [Q,R] = matlab.internal.math.deleteCol(Q, R, loc);
            ok = false;
            return;
        end
        cols(end+1) = j;
        ok = true;
    end

    function del_col(p)
        %   Remove column p of B and update (Q,R) by an internal Givens
        %   delete.  Q stays square, so no economy-size adjustment.
        R(:,p) = [];
        [Q,R] = matlab.internal.math.deleteCol(Q, R, p);
        cols(p) = [];
        if numel(qc) >= p
            qc(p) = [];
        end
    end

end
