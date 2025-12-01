%% run_test_qp_v1.m
% Random data, medium size, underdetermined.


%% Initialization
rng(1);
tol = 1e-8;

% Data
m = 1000; n = 2000;
A = randn(m,n); A = A/norm(A,2);
b = randn(m,1); c = randn(n,1);
t = 1.0;

use_nnls_di = true;

% 0. Generate a nontrivial feasible point.
x0 = zeros(n,1);
x0(1:m) = A(:,1:m)\b;

% 1. Inclusions algorithm
tic
disp('Running the qp_incl algorithm...')
[x_incl,~] = qp_incl_solver(A,b,c,t,x0,tol,use_nnls_di);
time_qp = toc;
disp(['Total time = ', num2str(time_qp)])

if(t>0)
    % 2. MATLAB's linprog or quadprog
    disp("Running MATLAB native quadprog algorithm...")
    tic
    x_prog = quadprog(t*eye(n),c,A,b);
    time_quadprog = toc;
    disp(["Total time = ", num2str(time_quadprog)])

    % Diagnostics
    disp(norm(x_incl-x_prog,'inf'))
end


% % Solve with MATLAB's native linprog
% tic
% [x_linprog,~,~,output] = linprog(c,A,b);
% time_lingprog = toc;


%% Notes:
% 1. qp_incl_solver is very fast.
% 2. Using a regularization path does not improve or slow down the QP
%    algorithm. Technically, it does mean we get it more or less for free?
%
% m=5000, n=10000, t = 1.0;   4.6s (qp_incl) vs 454.9s (quadprog)
% m=10000, n=5000, t = 0.1;   0.56s (qp_incl) vs 783.5s (quadprog)

% 3. Problems with t = 0. Fix!

% Probably the issue has to do with the fact that d may not be equal to
% zero when t = 0, so the nnls inclusions probably get stuck.