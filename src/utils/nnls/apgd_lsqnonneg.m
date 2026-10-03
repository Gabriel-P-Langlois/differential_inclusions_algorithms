function [xk,d,v] = apgd_lsqnonneg(A,b,tol,xk,v)
%   APGD_LSQNONNEG  Accelerated projected gradient descent for nonnegative
%                   least squares.
%
%   Solves
%
%   (1) min_{x in R^n}  (1/2)*||A*x - b||^2   subject to  x >= 0
%
%   by projected gradient descent with momentum and gradient restart:
%   Algorithm 4.2 of Liang, Luo, and Schoenlieb, "Improving FISTA: faster,
%   smarter, and greedier", with p = q = 1, r = 4, xi = 1.  The method is
%   iterative, not active set, and each iteration costs two matrix-vector
%   products, the dominant work.
%
% -------------------------------------------------------------------------
%   INPUTS
%       A             -   (m x n) matrix.
%       b             -   m-dimensional column vector.
%       tol           -   (Optional) Tolerance.  Default is derived from
%                         ||A||_F and the size of b.
%       xk            -   (Optional) Feasible start, xk >= 0.
%       v             -   (Optional) Warm start for the power iteration
%                         that estimates the Lipschitz constant L.
%
%   OUTPUTS
%       xk            -   Solution vector.
%       d             -   m-dimensional residual d = A*xk - b.
%       v             -   Singular vector of A at its largest singular
%                         value, to warm-start the next call.
%
% -------------------------------------------------------------------------
%   QUICK EXAMPLE
      % rng(1);
      % m = 200;  n = 2000;
      % A = randn(m,n);
      % b = randn(m,1);
      %
      % [~, ~, res_ml] = lsqnonneg(A, b);
      % [~, d]         = apgd_lsqnonneg(A, b, 1e-8);
      % fprintf('residual gap: %e\n', norm(d + res_ml, 'inf'))
%
% -------------------------------------------------------------------------


%%  Preliminary checks
% Check for an acceptable number of input arguments
if nargin < 2
    error('apgd_lsqnonneg: Not enough input arguments.')
end
if ~iscolumn(b)
    error('apgd_lsqnonneg: The data vector b is not a column vector.');
end

% Check matrix and right hand side vector inputs have appropriate sizes
[m,n] = size(A);
if size(b,1) ~= m
    error(['apgd_lsqnonneg: The number of rows of A does not match the' ...
        'length of the column vector b.'])
end

% Check if the optional input arguments have been supplied.
if nargin < 3 || isempty(tol)
    tol = min(1e-08,100*eps*max(norm(A, 'fro'), 1)*max(norm(b), 1));
end
if nargin < 4 || isempty(xk)
    if issparse(A)
        xk = sparse(n,1);
    else
        xk = zeros(n,1);
    end
end
if nargin < 5 || isempty(v)
    v = randn(n,1);
end

% Check if x is feasible
if(min(xk) < 0)
    xk = max(0,xk);
end


%% Algorithm
% Initialization
Atb = -A.'*b;
kmax = 50000;

% Estimate ||A||_{2} via MATLAB's normest function with warm start.
[L,v] = warm_normest(A,v,n);
tau = 1/L^2;

% Optimality threshold for the KKT (projected-gradient) test below, relative
% to the scale of the data term A'*b.
tol_pg = tol*max(1, norm(Atb));

% Main loop
xkm = xk;
tk = 1.0;
for k = 1:1:kmax
    % Iterations
    tkp = 0.5*(1 + sqrt(1 + 4*tk^2));
    betak = (tk-1)/tkp;
    yk = xk + betak*(xk-xkm);

    d = A*yk; 
    Atd = A.'*d;
    xkp = max(0,yk - tau*(Atd + Atb));

    % Check if restarting is needed
    if((yk-xkp).'*(xkp-xk) > 0)
        yk = xk;
        d = A*yk; 
        Atd = A.'*d;
        xkp = max(0,yk - tau*(Atd + Atb));
        tkp = 1.0;
    end

    % Check for convergence.  A short step is necessary but NOT sufficient:
    % from a warm start the first step can fall below the threshold with the
    % KKT residual still large, and this solver would return its own input.
    % Every caller warm-starts from the previous multipliers, so accept only
    % when the projected gradient certifies optimality as well.
    if(norm(xkp-xk) < tol*(1+norm(xkp)))
        g   = A.'*(A*xkp) + Atb;
        idx = (xkp <= 0);
        pg  = g;
        pg(idx) = min(0, g(idx));
        if(norm(pg) <= tol_pg)
            xk = xkp;
            break;
        end
    end

    % Update for the next iterate
    xkm = xk;
    xk  = xkp;
    tk  = tkp;
end

% Postprocess xk to improve its quality via an iterative solver
[xk,d] = postprocess_sol(A, b, xk, xk>tol );
end


%% Utility functions

function [L,v] = warm_normest(A,v,n)
%   WARM_NORMEST  Estimate the Lipschitz constant of A by power iteration,
%   MATLAB's normest tailored to accept a warm start.
e = norm(v);
if(e < 1e-06)
    v = randn(n,1);
    e = norm(v);
end

cnt = 0;
v = v/e;
e0 = 0;
while(abs(e-e0) > 1e-06*e)
    e0 = e;
    Av = A*v;
    normAv = norm(Av);
    v = A.'*Av;
    normv = norm(v);
    e = normv/normAv;
    v = v/normv;
    cnt = cnt + 1;
    if(cnt > 50000)
        disp("The power method for computing L did not converge.")
        break;
    end
end
L = e;
end

function [xk,d] = postprocess_sol(A,b,xk,eqcset)
%   POSTPROCESS_SOL  Polish the NNLS solution on its free face via lsqr.
    warning('off', 'MATLAB:lsqr:tooSmallTolerance')
    xk(~eqcset) = 0;
    [xk(eqcset),~] = lsqr(A(:,eqcset),b,eps,1000,[],[],xk(eqcset));
    d = A*xk-b;
    warning('on', 'MATLAB:lsqr:tooSmallTolerance')
end