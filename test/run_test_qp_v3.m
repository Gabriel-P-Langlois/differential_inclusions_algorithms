%% run_test_qp_v3.m
% Random data, medium size, underdetermined.


%% Initialization
rng(1);
tol = 1e-8;

% Data
m = 1000; n = 5000;
A = randn(m,n); A = A/norm(A,2);
b = ones(m,1); c = randn(n,1);
t = 0.0;

% 0. Generate a nontrivial feasible point.
x0 = zeros(n,1);

% 1. Inclusions algorithm
tic
disp('Running the qp_incl_qr.m algorithm...')
[x_incl,~] = qp_incl_qr_solver(A,b,c,t,tol,x0);
time_qp = toc;
disp(['Total time = ', num2str(time_qp)])
disp(' ')


% 1.5 Inclusions algorithm (no initial point given).
tic
disp('Running the qp_incl_qr.m algorithm with no x0')
[x_incl2,~] = qp_incl_qr_solver(A,b,c,t,tol);
time_qp2 = toc;
disp(['Total time = ', num2str(time_qp2)])
disp(' ')

% % 2. MATLAB Quadprog or linprog
% if(t>0)
%     % quadprog, default solver.
%     disp("Running MATLAB's native quadprog algorithm (default solver)")
%     tic
%     x_quadprog = quadprog(t*eye(n),c,A,b);
%     time_quadprog = toc;
%     disp(["Total time = ", num2str(time_quadprog)])
%     disp(norm(x_incl-x_quadprog,'inf'))
%     disp(' ')
% 
%     % quadprog, active set solver.
%     options = optimoptions('quadprog','Algorithm','active-set');
%     disp("Running MATLAB's native quadprog algorithm (active-set)")
%     tic
%     x_quadprog_as = quadprog(t*eye(n),c,A,b,[],[],[],[],x0,options);
%     time_quadprog_as = toc;
%     disp(["Total time = ", num2str(time_quadprog_as)])
%     disp(norm(x_incl-x_quadprog_as,'inf'))
%     disp(' ')
% 
% else
%     % linprog, default sovler
%     disp("Running MATLAB's native linprog algorithm (default solver)")
%     tic
%     x_linprog = linprog(c,A,b);
%     time_linprog = toc;
%     disp(["Total time = ", num2str(time_linprog)])
%     disp(norm(x_incl-x_linprog,'inf'))
%     disp(' ')
% 
%     % linprog, interior point solver
%     options = optimoptions('linprog','Algorithm','interior-point');
%     disp("Running MATLAB's native linprog algorithm " + ...
%         "(interior point solver)")
%     tic
%     x_linprog_ip = linprog(c,A,b,[],[],[],[],options);
%     time_linprog_ip = toc;
%     disp(["Total time = ", num2str(time_linprog_ip)])
%     disp(norm(x_incl-x_linprog_ip,'inf'))
%     disp(' ')
% end

%% Notes:
%%% t > 0 (quadratic programming)
% With (m,n) = (1000,5000) and t = 1.0:
%   - qp_incl_qr.m code takes about 1.37s.
%   - quadprog (MATLAB's native code \w IP) takes about 23.44s.
%     The absolute linf error is norminf(x_incl-xquadprog) = 1.2913e-06.
%   - quadprog (MATLAB's native code \w active-set) takes about 440.24s.
%     The absolute linf error is norminf(x_incl-xquadprog_as) = 7.7716e-15.
%
%
% With (m,n) = (5000,1000) and t = 1.0:
%   - qp_incl_qr.m code takes about 0.0185s.
%   - quadprog (MATLAB's native code \w IP) takes about 53.53s.
%     The absolute linf error is norminf(x_incl-xquadprog) = 9.9920e-16.
%   - quadprog (MATLAB's native code \w active-set) takes about 0.1734s.
%     The absolute linf error is norminf(x_incl-xquadprog_as) = 8.8818e-16.



%%% t = 0 (linear programming) %%%
% With (m,n) = (1000,5000) and t = 0.0:
%   - qp_incl_qr.m code takes about 8.44s.
%     No feasible solution found.
%
%   - linprog (MATLAB's native code \w interior-point) takes ~ 12.53s.
%     No feasible solution found.
%
%   - linprog (MATLAB's native code + default solver) takes ~ 79.75.
%     No feasible solution found.
%
%
% With (m,n) = (5000,1000) and t = 0.0:
%   - qp_incl_qr.m code takes about 16.9742s. 
%     Feasible solution found.
%
%   - linprog (MATLAB's native code \w interior-point) takes ~ 77.01317s.
%     The absolute linf error is norminf(x_incl-xlinprog) = 3.173e-09.
%
%   - linprog (MATLAB's native code + default solver) takes ~ 40.0096s.
%     The absolute linf error is norminf(x_incl-xlinprog_ip) = 0.0025.
