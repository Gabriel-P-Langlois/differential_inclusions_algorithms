function [xplus,num_iters] = feas_fista_solver(x,t,A,b,...
    tau,max_iters,tol,min_iters)
% feas_fista_solver     Computes the primal and dual solution of the
%                       FEAS-r problem
%                       min_{x \in \Rn} \frac{1}{2}\normsq{max(0,Ax-b)} 
%                                       + t||x||_1
%                       within a certain tolerance level using the
%                       forward-backward algorithm.
%
%   Input
%       x           -   n dimensional col vector. This is used as a warm
%                       start for the fista algorithm, typically using a
%                       primal solution at a different parameter t.
%
%       t           -   positive hyperparameter of the LASSO problem.
%       A           -   m by n design matrix of the LASSO problem.
%       b           -   m dimensional col data vector of the LASSO problem.
%       tau         -   positive number based on the matrix A
%       max_iters   -   positive number
%       tol         -   small positive number (e.g., 1e-08)
%       min_iters   -   positive number
%
%   Output
%       xplus       -   Primal solution to the FEAS problem at 
%                       parameter t with data A and b.
%       num_iters   -   Number of iterations taken to calculate the
%                       primal and dual solutions.


%% Algorithm

% Counter for the iterations
num_iters = 0;

% Iterations
for k=1:1:max_iters
    num_iters = num_iters + 1;
    
    % FISTA Variable update
    tmp1 = tau*max(0,A*x - b);
    tmp2 = x - A.'*tmp1;
    
    % Proximal calculation
    xplus = l1_prox_operator(tmp2,tau*t);
    
    % Check for convergence
    if(num_iters >= min_iters)
        %stop = norm((-pplus.'*A).',inf) <= t*(1 + tol);
        stop = norm(xplus-x,2) <= tol*norm(xplus);
        if(stop || (num_iters >= max_iters))
            break;
        end
    end
    
    % Increment
    x = xplus;
end
end