%%% check_speed_1 %%%
% This simply checks the speed of the DI algorithm for FEAS.

%% Options
% Global parameters
tol = 1e-8;
rng(1);

% Data
m = 1000;
n = 2000;
A = randn(m,n); A = A./norm(A,2);
b = randn(m,1); b = b./norm(b,2);

% Initial values
x0 = zeros(n,1);
p0 = max(0,-b);
tmax = max(norm(A.'*p0,'inf'),max(p0));
p0 = p0/tmax;


%% Test 1 -- QR FEAS algorithm
tic
[x_di_2,p_di_2,v_di_2,d_di_2,count_nnls_2] = ...
       feas_di_QR_solver(A,b,0,p0,tol);
time_feas = toc;
disp(['Time for the differential inclusions algorithm: ', num2str(time_feas)])


%% Test 2 -- MATLAB's linprog (simplex method)
tic
[y_linprog,~,~,output] = linprog(ones(2*n,1),[A,-A],b,[],[],zeros(2*n,1),[]);
x_linprog = y_linprog(1:n) - y_linprog(n+1:2*n);
time_linprog_simplex = toc;
disp(['Time for the simplex algorithm (dual-simplex-highs): ',num2str(time_linprog_simplex)])


%% Output
disp(norm(x_linprog-x_di_2))