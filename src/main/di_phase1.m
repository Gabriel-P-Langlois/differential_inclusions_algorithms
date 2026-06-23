function [x, flag] = di_phase1(A, b, tol, maxit, nnls_solver)
%   DI_PHASE1           Differential inclusions approach to finding a
%                       feasible point for a linear inequality system, or
%                       certifying infeasibility.
%
%   Given a constraint matrix A and data vector b, this function determines
%   whether the system
%
%   (1)     A*x <= b
%
%   is feasible and, if so, returns a feasible point x.
%
%   The approach reduces the feasibility problem to the Phase I linear
%   program
%
%   (2) min_{x in R^n, t in R} t   subject to A*x - t*1 <= b,  -t <= 0,
%
%   which is solved via di_lp. The system (1) is feasible if and only if
%   the optimal value of (2) is at most zero.
%
% -------------------------------------------------------------------------
%   INPUTS
%       A           -   (m x n)-dimensional matrix
%       b           -   m-dimensional col data vector
%       tol         -   (Optional) Tolerance. Default is derived from the
%                       Frobenius norm of A and the relative size of b.
%       maxit       -   (Optional) Maximum number of outer iterations for
%                       di_lp. Default is max(1e6, 100*(n+1)).
%       nnls_solver -   (Optional) Function handle to an NNLS solver,
%                       passed directly to di_lp. Default is
%                       @epgd_lsqnonneg.
%
%   OUTPUTS
%       x           -   n-dimensional feasible point satisfying A*x <= b,
%                       or empty if infeasible or if the outer loop
%                       reached maxit.
%       flag        -   Returns 1 if a feasible point is found, 0 if the
%                       system is infeasible, and -1 if the outer loop of
%                       di_lp reached maxit without converging.
%
% -------------------------------------------------------------------------
%   QUICK EXAMPLE
      % % Generate the data
      % m = 500;
      % n = 1000;
      % x0 = randn(n, 1);
      % A  = randn(m, n);
      % s  = abs(randn(m, 1)) + 1;
      % b  = A * x0 + s;
      %
      % [x, flag] = di_phase1(A, b);
      %
      % tol = min(1e-08, 100*eps*max(norm(A,'fro'),1)*max(norm(b),1));
      % assert(flag == 1)
      % assert(all(A*x <= b + tol))
%
% -------------------------------------------------------------------------


%% Preliminary checks
if nargin < 2
    error('di_phase1: Not enough input arguments.')
end
if ~iscolumn(b)
    error('di_phase1: The data vector b is not a column vector.');
end

[m, n] = size(A);
if size(b, 1) ~= m
    error(['di_phase1: The number of rows of A does not match the ' ...
        'length of the column vector b.'])
end

if nargin < 3 || isempty(tol)
    tol = min(1e-08, 100*eps*max(norm(A, 'fro'), 1)*max(norm(b), 1));
end
if nargin < 4 || isempty(maxit)
    maxit = max(1e6, 100*(n + 1));
end
if nargin < 5 || isempty(nnls_solver)
    nnls_solver = @apgd_lsqnonneg;
end


%% Build Phase I augmented LP
% Variables: z = [x; t] in R^{n+1}.
%   Constraints: [A, -1; 0', -1] * [x; t] <= [b; 0]
%   Objective:   c' = [0_n; 1]
A_aug = [A, -ones(m, 1); zeros(1, n), -1];
b_aug = [b; 0];
c_aug = [zeros(n, 1); 1];

% Starting point: x0 = 0, t0 chosen so all Phase I residuals >= 1.
t0  = max(0, -min(b)) + 1;
z0  = [zeros(n, 1); t0];


%% Solve Phase I LP via di_lp
[z, ~, tval, di_flag] = di_lp(A_aug, b_aug, c_aug, z0, tol, maxit, ...
    nnls_solver);

%% Interpret result
if di_flag == 1
    if tval <= tol
        x    = z(1:n);
        flag = 1;
    else
        x    = [];
        flag = 0;
    end
elseif di_flag == -1
    x    = [];
    flag = -1;
else
    error(['di_phase1: di_lp returned an unexpected flag. The Phase I ' ...
        'LP should always be feasible and bounded.'])
end


end
