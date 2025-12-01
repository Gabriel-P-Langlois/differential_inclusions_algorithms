function Vdual = feas_dual_obj_fun(p,t,A,b)
% FEAS_DUAL_OBJ_FUN.m    Computes the dual objective function of
%                        the FEAS problem at fixed t>=0

tol = 1e-8;

Vdual = 0;
term_1 = p.'*b;
term_2 = (0.5*t)*(norm(p,2)^2);
term_3 = norm(A.'*p,inf);
term_4 = min(0,p);

if((-1 - tol) <= term_3 && (term_3 <= 1+tol) && (norm(term_4,'inf') < tol))
    Vdual = term_1 + term_2;
else
    Vdual = inf;
    warning("Infeasible dual point!")
end