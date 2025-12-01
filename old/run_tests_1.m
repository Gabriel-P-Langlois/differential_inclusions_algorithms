%%%%% run_tests_1 %%%%%
% These are a series of 6 tests to check whether the NNLS solvers,
% primal and dual objective functions calculations, and FEAS_DI algorithm
% work properly. The last test pertains to an early form of the ``greedy"
% algorithm found in our first paper.

% Global parameters
tol = 1e-8;
rng(1);

% Data
m = 200;
n = 400;
A = randn(m,n); A = A/norm(A,2);
b = randn(m,1); b = b/norm(b,2);


%% Test 1 -- Nonnegative least-squares solver
% Compute the NNLS solution using 1) MATLAB's native lsqnonneg
% function and 2) the method of hinges
x_matlab = lsqnonneg(A,b);
x_hinge = hinge_lsqnonneg(A,b,tol);

% Compute their respective residuals Ax-b and the norm of their residuals
d_matlab = A*x_matlab - b;
d_hinges = A*x_hinge - b;

% Output
disp(' ')
disp('--- Test 1: NNLS solvers ---')
test_1 = norm(d_matlab - d_hinges) < tol;
if(test_1)
    disp("Test 1 (NNLS): Passed!")
else
    error("Test 1 (NNLS): Failed! Please fix me...")
end
disp(' ')



%% Test 2 -- Primal and dual objective functions
% Check that feas_primal_obj_fun works correctly.
x0 = zeros(n,1);
p0 = max(0,-b);
tmax = max(norm(A.'*p0,'inf'),max(p0));
p0 = p0/tmax;

% Primal
x = zeros(n,1);
case_1 = feas_primal_obj_fun(x,tmax,A,b);
case_2 = feas_primal_obj_fun(x,0,A,b);
case_3 = feas_primal_obj_fun(x,0,A,ones(m,1));

% Dual
case_4 = feas_dual_obj_fun(p0,tmax,A,b);
case_5 = feas_dual_obj_fun(randn(m,1),1,A,b);
case_6 = feas_dual_obj_fun(p0,1,A,b);

% Output
disp('--- Test 2: Primal and dual objective functions ---')
disp("Test 2 (FEAS_PRIMAL_OBJ_FUN): " + ...
    "The numbers below should be -case_4/inf/0.")
disp([case_1,case_2,case_3])
disp("Test 2 (FEAS_DUAL_OBJ_FUN): " + ...
    "The numbers below should be -case_1/inf/finite.")
disp([case_4,case_5,case_6])



%% Test 3 -- Forward--Backward method for the regularized FEAS problem
tol_fwb = 1e-08;
t_fwb = tmax/2;
max_iters = 50000;
min_iters = 1000;
L22 = svds(A,1)^2;
tau = 1/L22;

tic
[x_fwb,num_iters_fwb] = feas_fista_solver(x0,t_fwb,A,b,...
    tau,max_iters,tol_fwb,min_iters);
time_fwb = toc;

p_fwb = max(0,A*x_fwb-b)/t_fwb;
feas_primal_fwb = feas_primal_obj_fun(x_fwb,t_fwb,A,b);
feas_dual_fwb = feas_dual_obj_fun(p_fwb,t_fwb,A,b);
rel_diff_fwb = (feas_primal_fwb + feas_dual_fwb)/feas_primal_fwb;

% Output
disp('--- Test 3: Forward-Backward method with t = tmax/2 ---')
disp(['Time taken for the FWB method with tolerance ',num2str(tol_fwb),...
    ': ',num2str(time_fwb),' seconds.'])
disp(['Primal objective function: ', ...
    num2str(feas_primal_fwb)])
disp(['Dual objective function: ', ...
    num2str(feas_dual_fwb)])
disp(['Relative difference between the primal and dual objective functions: ', ...
    num2str(rel_diff_fwb)])



%% Test 4 -- Differential inclusions at t>0
t_di = t_fwb;
[x_di,p_di,v_di,d_di,count_nnls] = feas_di_solver(A,b,t_di,p0,tol);

% Output
disp(' ')
disp('--- Test 4: Differential inclusions at t = tmax/2 ---')
feas_primal_di = feas_primal_obj_fun(x_di,t_di,A,b);
feas_dual_di = feas_dual_obj_fun(p_di,t_di,A,b);
rel_diff_di = (feas_primal_di + feas_dual_di)/abs(feas_dual_di);
disp(['Primal objective function: ', ...
    num2str(feas_primal_di)])
disp(['Dual objective function: ', ...
    num2str(feas_dual_di)])
disp(['Relative difference between the primal and dual objective functions: ', ...
    num2str(rel_diff_di)])
disp(['Computing the linf norm of the residual vector M*w - (b + t*p): ',...
    num2str(norm(d_di,'inf'))])
disp(['Computing the linf of max(0,A*x - b): ', ...
    num2str(norm(max(0,A*x_di - b),'inf'))])


%% Test 5 -- Differential inclusions at t=0 + comparison \w linprog
tic
[x_di_0,p_di_0,v_di_0,d_di_0,count_nnls_0] = feas_di_solver(A,b,0,p0,tol);
time_di_0 = toc;

tic
[y_linprog,~,~,output] = linprog(ones(2*n,1),[A,-A],b,[],[],zeros(2*n,1),[]);
x_linprog = y_linprog(1:n) - y_linprog(n+1:2*n);
time_lingprog = toc;

% Output
disp(' ')
disp('--- Test 5: Differential inclusions + linprog at t = 0 ---')
feas_primal_di_0 = feas_primal_obj_fun(x_di_0,0,A,b);
feas_dual_di_0 = feas_dual_obj_fun(p_di_0,0,A,b);
rel_diff_di_0 = (feas_primal_di_0 + feas_dual_di_0)/abs(feas_dual_di_0);
disp(['Primal objective function: ', ...
    num2str(feas_primal_di_0)])
disp(['Dual objective function: ', ...
    num2str(feas_dual_di_0)])
disp(['Relative difference between the primal and dual objective functions: ', ...
    num2str(rel_diff_di_0)])
disp(['Computing the linf norm of the residual vector M*w - (b + t*p): ',...
    num2str(norm(d_di_0,'inf'))])
disp(['Computing the linf of max(0,A*x - b): ', ...
    num2str(norm(max(0,A*x_di_0 - b),'inf'))])

feas_primal_linprog = feas_primal_obj_fun(x_linprog,0,A,b);
disp(['l2-norm between primal obj. of solutions found ' ...
    'with differential inclusion and linprog: ',...
    num2str(norm(feas_primal_linprog - feas_primal_di_0))]);


%% Test 6 -- Dual greedy method
[x_g, y_g, p_g, v_g, b_g, w_g, d_g, count_lsq] = feas_di_greedy(A,b,tol);

disp(' ')
disp('--- Test 6: Greedy method ---')
feas_primal_g = feas_primal_obj_fun(x_g,0,A,b_g);
feas_dual_g = feas_dual_obj_fun(p_g,0,A,b_g);
rel_diff_g = (feas_primal_g + feas_dual_g)/abs(feas_dual_g);
disp(['Primal objective function: ', ...
    num2str(feas_primal_g)])
disp(['Dual objective function: ', ...
    num2str(feas_dual_g)])
disp(['Relative difference between the primal and dual objective functions: ', ...
    num2str(rel_diff_g)])
disp(norm(max(0,A*x_g-b_g),'inf'))


%% Test 7 -- LP algorithm on random data
disp('Test 7')
c = -ones(n,1);
p0 = max(0,-b);
t0 = abs(min(A.'*p0));
p0 = p0 / t0;

tic
[x_lp_2,v_lp_2,p_lp_2,res_2,count_nnls_2] = lp_di_solver(A,b,c,p0,tol);
time_lp_di_2 = toc;

tic
[x_linprog_2,~,~,output] = linprog(-c,A,b,[],[],zeros(n,1),[]);
time_lingprog_2 = toc;


%% Notes:
% 1)  The greedy method does not appear to work like in the OG paper.
% But at least the algorithm works -- this means we can extend the method
% of differential inclusions to LP and quadratic conic programming in
% general. This is a paper on its own...