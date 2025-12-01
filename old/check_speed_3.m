%%% check_speed_3 %%%
% This code checks the speed of linprog and the DI algorithm
% on the Klee--Minty cube.


%% Options
% Global parameters
tol = 1e-8;
d = 12;

% Generate the Klee-Monty cube
A = eye(d,d);
for i=2:1:d
    for j=1:1:i-1
        A(i,j) = 2^(i-j+1);
    end
end

b = zeros(d,1);
c = zeros(d,1);
for i=1:1:d
    b(i) = 5^i;
    c(i) = 2^(d-i);
end


% Initial values
x0 = zeros(d,1);
p0 = ones(d,1);


%% Test 1 -- DI algorithm with QR and random data
% DI algorithm with QR decomposition on the primal problem
tic
[p_lp_qr,~,x_lp_qr,res,count_nnls,linsolve] = lp_di_QR_solver(-A.',-c,-b,x0,tol);
time_lp_di_QR = toc;

% DI algorithm with QR decomposition on the dual problem
tic
[x_dual,~,p_dual,res_dual,count_nnls_dual,linsolve_dual] = ...
    lp_di_QR_solver(A,b,c,p0,tol);
time_dual = toc;

% linprog function
tic
[x_linprog,~,~,output] = linprog(-c,A,b,[],[],zeros(d,1),[]);
time_lingprog = toc;
disp(norm(x_linprog-p_lp_qr))