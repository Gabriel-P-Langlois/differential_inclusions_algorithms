%%%%% run_tests_2 %%%%%
% Global parameters
tol = 1e-8;
rng(1);

% Data
m = 200;
n = 400;
A = randn(m,n);
b = randn(m,1);

% Initial values
x0 = zeros(n,1);
p0 = max(0,-b);
tmax = max(norm(A.'*p0,'inf'),max(p0));
p0 = p0/tmax;


%% Test 1 -- Accuracy checks
% Check how accurate the DI algorithm is vs the FWB algorithm.
% No regularization path is taken.

% Regularization path
t = tmax*(0.025:0.025:1.0);
kmax = length(t);

% FWB algorithm -- starting from x0 = 0 each time.
min_iters = 1000;
max_iters = 50000;
L22 = svds(A,1)^2;
tau = 1/L22;
x_fwb = zeros(n,kmax);
dual_obj_fwb = zeros(kmax,1);

tic
for k=kmax-1:-1:1
    [x_fwb(:,k),num_iters_fwb] = feas_fista_solver(x_fwb(:,k+1),t(k),A,b,...
    tau,max_iters,tol,min_iters);
    dual_obj_fwb(k) = feas_primal_obj_fun(x_fwb(:,k),t(k),A,b);
end
time_fwb = toc;

% DI algorithm -- starting from p = p0 each time.
x_di = zeros(n,kmax);
p_di = zeros(m,kmax); p_di(:,kmax) = p0;
v_di = zeros(m,kmax);
d_di = zeros(m,kmax);
count_nnls = zeros(kmax,1);
dual_obj_di = zeros(kmax,1);
tic
for k=kmax-1:-1:1
   [x_di(:,k),p_di(:,k),v_di(:,k),d_di(:,k),count_nnls(k)] = ...
       feas_di_solver(A,b,t(k),p_di(:,k+1),tol);
   dual_obj_di(k) = feas_primal_obj_fun(x_di(:,k),t(k),A,b);
end
time_di = toc;

% Plot the results
RGB = orderedcolors("gem");
f1 = figure(1);
loglog(t/tmax,dual_obj_fwb,'o','DisplayName','FWB algorithm',...
    'MarkerFaceColor',RGB(2,:),'MarkerEdgeColor','k')
hold(gca,'on')
loglog(t/tmax,dual_obj_di,'x','DisplayName','DI algorithm',...
    'MarkerFaceColor',RGB(1,:),'MarkerEdgeColor','k')
title('Primal objective function evaluated at an optimal solution', ...
    'fontsize',14)
xlabel('t/tmax')
ylabel('Primal objective function: ||x||_{1} + (0.5/t)||max(0,Ax-b)||_{2}^{2}.')


%%%
% %%%%%%%%%%
    % % 1. Plot the dual function V(p^{s}(t,b)) as a function of t
    % % %%%%%%%%%%
    % f1 = figure(1);
    % loglog(t/t0,rel_err_objd_homotopy_alg1,'o',...
    %     'DisplayName','Algorithm 2 (homotopy)',...
    %     'MarkerFaceColor',RGB(2,:),'MarkerEdgeColor','k')
    % hold(gca,'on')
    % loglog(t(1:end-1)/t0,rel_err_objd_glmnet_alg1(1:end-1),...
    %     'square','DisplayName','glmnet',...
    %     'MarkerFaceColor',RGB(1,:),'MarkerEdgeColor','k')    
    % loglog(t(1:end-1)/t0,rel_err_objd_mlasso_alg1(1:end-1),...
    %     'diamond','DisplayName','mlasso',...
    %     'MarkerFaceColor',RGB(4,:),'MarkerEdgeColor','k')
    % loglog(t(1:end-1)/t0,rel_err_objd_fista_alg1(1:end-1),...
    %     '^','DisplayName','fista',...
    %     'MarkerFaceColor',RGB(3,:),'MarkerEdgeColor','k')
    % 
    % title('Relative error of the dual objective function w.r.t. Algorithm 1',...
    %     'interpreter','latex','fontsize',14)
    % xlabel('t/$||A^{\top}b||_{\infty}$','interpreter','latex','fontsize',14)
    % ylabel(['$|(V(p_{\mathrm{X}}^{s})(t,b) - V(p_{\mathrm{alg1}}^{s}(t,b))|' ...
    %     '/|V(p_{\mathrm{alg1}}^{s}(t,b))|$'],'interpreter','latex','fontsize',14)
    % xlim([10^-3 1])
    % ylim([10^-19 10^0])
    % lgd = legend('Location','southeast');
    % fontsize(lgd,14,'points')
    % grid on
    % set(gca, 'xdir', 'reverse')
    % set(gca,'xminorgrid','on','yminorgrid','off')   
%%%