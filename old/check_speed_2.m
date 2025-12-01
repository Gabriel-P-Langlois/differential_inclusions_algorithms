%%% check_speed_2 %%%
% This simply checks the speed of the DI algorithm for LP.


%% Options
% Global parameters
tol = 1e-8;
rng(1);

% Data
m = 200;
n = 400;
A = randn(m,n); %A = A./norm(A,2);
b = randn(m,1); %b = b./norm(b,2);

% Initial values
p0 = max(0,-b);
c = A.'*p0;


%% Test 1 -- DI algorithm with QR and random data
% DI algorithm with QR
tic
[x_lp_qr,v_lp_qr,p_lp_qr,res,count_nnls,linsolve] = lp_di_QR_solver(A,b,c,p0,tol);
time_lp_di_QR = toc;

% linprog function
tic
[x_linprog,~,~,output] = linprog(-c,A,b,[],[],zeros(n,1),[]);
time_lingprog = toc;

% Comparisons
obj1 = c.'*x_lp_qr;
obj2 = c.'*x_linprog;
disp(abs(obj1-obj2))