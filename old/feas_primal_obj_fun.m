function Vprimal = feas_primal_obj_fun(x,t,A,b)
% FEAS_PRIMAL_OBJ_FUN.m  Computes the primal objective function of
%                        the FEAS problem at fixed t>=0

term_1 = norm(max(0,A*x - b))^2;
term_2 = norm(x,1);
tol = 1e-8;

if(t>0)
    Vprimal = (0.5/t)*term_1 + term_2;
elseif(t==0)
    if(term_1 < tol)
        Vprimal = term_2;
    else
        Vprimal = inf;
        warning("Infeasible primal point!")
    end
else
    error("Error: The hyperparameter t is negative and invalid.")
end
end