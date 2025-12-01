%% run_tests_l1feas.m
% This script tests the l1feas_incl_solver.m and l1feas_incl_qr_solver.m
% function.


%% Initialization
rng(1);
tol = 1e-8;

%% Accuracy + speed test I: Underdetermined system, small size.
% Data
m = 50; n = 100;
A = randn(m,n); A = A/norm(A,2);
b = randn(m,1); c = randn(n,1);

tic
disp("Running l1feas solver (no QR)")
x_l1feas = l1feas_incl_solver(A,b,tol,false);
time_l1feas = toc;
disp(['Total time = ', num2str(time_l1feas)])

tic
disp("Running l1feas solver (\w QR)")
[x_l1feas_qr,~,~,~] = l1feas_incl_qr_solver(A,b,tol);
time_l1feas_qr = toc;
disp(['Total time = ', num2str(time_l1feas_qr)])

disp(['linf norm between solutions: ', ...
    num2str(norm(x_l1feas_qr - x_l1feas,'inf'))])
disp(' ')
disp('-----')
disp(' ')


%% Accuracy + speed test II: Overdetermined system, small size.
% Note: There should not be a feasible point.
% Data
m = 200; n = 100;
A = randn(m,n); A = A/norm(A,2);
b = randn(m,1); c = randn(n,1);

tic
disp("Running l1feas solver (no QR)")
x_l1feas = l1feas_incl_solver(A,b,tol,false);
time_l1feas = toc;
disp(['Total time = ', num2str(time_l1feas)])

tic
disp("Running l1feas solver (\w QR)")
[x_l1feas_qr,Q,R,eqset] = l1feas_incl_qr_solver(A,b,tol);
time_l1feas_qr = toc;
disp(['Total time = ', num2str(time_l1feas_qr)])

disp(['linf norm between solutions: ', ...
    num2str(norm(x_l1feas_qr - x_l1feas,'inf'))])

%% Notes:
