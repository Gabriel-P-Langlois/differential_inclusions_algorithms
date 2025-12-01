%% run_test_qp_v1.m
% Random data, medium size.
% Test the l1 feature.


%% Initialization
rng(1);
tol = 1e-8;

% Data
m = 500; n = 1000;
A = randn(m,n); A = A/norm(A,2);
b = randn(m,1); c = randn(n,1);
use_nnls_di = false;
t = 0.0;

tic
[x_incl,~] = qp_incl_solver(A,b,c,t,tol,use_nnls_di);
time_incl = toc;
disp(["Total time = ", num2str(time_incl)])

% 2. MATLAB Quadprog or linprog
if(t>0)
    % quadprog
    disp("Running MATLAB's native quadprog algorithm...")
    tic
    x_quadprog = quadprog(t*eye(n),c,A,b);
    time_quadprog = toc;
    disp(["Total time = ", num2str(time_quadprog)])

    disp(norm(x_incl-x_quadprog,'inf'))
else
    % linprog
    disp("Running MATLAB's native linprog algorithm...")
    tic
    x_linprog = linprog(c,A,b);
    time_linprog = toc;
    disp(["Total time = ", num2str(time_linprog)])

    disp(norm(x_incl-x_linprog,'inf'))
end