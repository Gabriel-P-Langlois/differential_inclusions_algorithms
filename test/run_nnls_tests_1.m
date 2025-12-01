%% RUN_NNLS_TESTS_1
% Random Gaussian data;
% Small data sizes (to include the hinge method).


%% Initialization
% Global parameters
tol = 1e-8;
rng(1);

% Data
m = 400; n = 800;
A = randn(m,n); A = A/norm(A,2);
b = randn(m,1); b = b/norm(b,2);
x0 = zeros(n,1);


%% Methods
% TEST 1.1 Differential inclusions method
tic
[x_di,p_di,iters] = nnls_inclusions(A,b,x0);
time_di = toc;
disp(['Number of iterations:', num2str(iters)])


% TEST 1.2 MATLAB's default quadprog function (interior point method)
tic
x_quadprog_0 = quadprog(A.'*A,-A.'*b,[],[],[],[],zeros(n,1),[]);
time_quadprog_0 = toc;


% TEST 1.3 MATLAB's trust-region-reflective algorithm
options_1 = optimoptions('quadprog','Algorithm','trust-region-reflective');
tic
x_quadprog_1 = quadprog(A.'*A,-A.'*b,[],[],[],[],zeros(n,1),[],zeros(n,1),options_1);
time_quadprog_1 = toc;


% TEST 1.4 Method of Hinges (Vanilla implementation)
tic
x_hinge = nnls_hinge(A,b,tol,false(n,1));
time_hinge = toc;

% TEST 1.5 Projected gradient descent
tic
kmax = 50000;
eta = 2/(norm(A)^2);
[x_pgd,p_pgd] = nnls_pgd(A,b,x0,kmax,tol,eta);
time_PGD = toc;

% Calculate dual vectors
p_quadprog_0 = A*x_quadprog_0 - b;
p_quadprog_1 = A*x_quadprog_1 - b;
p_hinge = A*x_hinge - b;

disp('linf norm sol difference between DI and quadprog')
disp(norm(p_di-p_quadprog_0,'inf'))
disp(' ')

disp('linf norm sol difference between DI and method of hinges')
disp(norm(p_di-p_hinge,"inf"))
disp(' ')


%disp(norm(p_di-p_quadprog_1,'inf'))
%disp(norm(p_di-p_pgd,'inf'))