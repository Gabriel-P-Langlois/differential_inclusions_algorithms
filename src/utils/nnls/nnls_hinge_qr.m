function [x,d,eqset,Q,R,num_linsolve] = ...
    nnls_hinge_qr(A,Q,R,b,eqset,opts,tol)
% NNLS_HINGE_QR   This function computes the nonnegative LSQ problem
%                       min_{x>=0} ||A*x-b||_{2}^{2}
%                       using the ``Method of Hinges" presented in
%                       ``A Simple New Algorithm for Quadratic Programming 
%                       with Applications in Statistics" by Mary C. Meyers.
%
%                   1) This version of the Method of Hinges starts with
%                   the full active set and maintains the QR
%                   decomposition of A as we remove elements from the
%                   active set or add elements to the active set.
%
%                   2) This function works with dense matrices only.
%
% INPUT:
%           A           -   (m x n)-dimensional dense matrix A
%           Q           -   (m x n)-dimensional, non-economic
%                           orthogonal matrix obtained from the QR
%                           decomposition [Q,R] = qr(A).
%           R           -   (n x n)-dimensional, economic form of the
%                           upper triangular matrix obtained from the QR
%                           decomposition [Q,R] = qr(A,"econ","vector")
%                           min_{x \in \Rn} ||Ax - b||_{2}^{2}.
%           b           -   m-dimensional col data vector.
%           eqset      -   equicorrelation set of the BPDN problem
%                           NOTE: Used to improve performance, does not
%                           actually recompute the equicorrelation set.
%           opts        -   opts field for the linsolve function.
%           tol         -   small number specifying the tolerance
%                           (e.g., 1e-08).
%
% OUTPUT:
%           x   -   n-dimemsional solution vector to the NNLS problem.
%           d   -   m-dimensional col residual vector d = A*x-b
%           eq_set      -   Updated equicorrelation set of the BPDN problem
%           Q   -   Updated orthogonal matrix from the QR decomposition
%           R   -   Updated upper triangular matrix from the QR
%                   decomposition
%           num_linsolve     - Number of times linsolve is called


%% Initialization
if(issparse(A))
    error("No support for sparse matrices")
end
[~, n] = size(A);
active_set = true(n,1);
tmp = Q.'*b;


%% Check if the LSQ is admissible and identify potential issues
[u,r1] = linsolve(R,tmp,opts);
num_linsolve = 1;
    
% Check if the system is singular. If so, avoid solving it with QR.
if(r1 < tol)
   [x,d] = hinge_lsqnonneg(A,b,tol);
   active_set(x==0) = false;
   [Q,R] = qr(A(:,active_set));
   ind = find(eqset);
   eqset(ind(~active_set)) = 0;
   return;
end

% Check whether the LSQ solution is admissible; is so, return it.
if(min(u) > -tol)
    x = u; 
    d = A*x-b;
    return;
end


%% Compute the NNLS solutions via Meyers algorithm
while(true)
    remove_update = false;
    insert_update = false;

    % Check if the primal constraint is violated. If so, mark
    % the violating element. Else, compute the residual b - A*u.
    if(min(u) <= -tol)  
        x = zeros(n,1); 
        x(active_set) = u;
        [~,I] = min(x);
        active_set(I) = false;
        remove_update = isscalar(I);
    else    
        tmp = R*u;
        theta = Q*tmp;
        theta = b - theta;
        Atop_times_rho = (theta.'*A);
        [val,I] = max(Atop_times_rho);

        % Check if the dual constraint holds. If so, EXIT. Otherwise,
        % mark the corresponding columns to be added to the active set.
        if(val < tol)
            break;
        else
            active_set(I) = true;
            insert_update = isscalar(I);
        end
    end

    % Update the QR decomposition.
    if(remove_update)
        % Exact element removed from eq_set and its column
        [~, J] = min(u);

        % Call MATLAB's [Q,R] = qrdelete(Q,R,J); without overhead
        R(:,J) = [];
        [Q,R]=matlab.internal.math.deleteCol(Q,R,J);
    elseif(insert_update)
        % Extract new element added to eq_set and its column.
        [~, J] = max(Atop_times_rho);
        col = A(:,J);
        loc = find(find(active_set) == J);

        % Call MATLAB's [Q,R] = qrinsert(Q,R,loc,col) without overhead
        [~,nr] = size(R);
        R(:,loc+1:nr+1) = R(:,loc:nr);
        R(:,loc) = Q'*col;
        [Q,R] = matlab.internal.math.insertCol(Q,R,loc);
    else
        % Multiple column deletes or inserts not supported.
        [Q,R] = qr(A(:,active_set));
    end
    
    % Compute LSQ solution to Au = b with the QR decomposition A=Q*R.
    tmp = Q.'*b;
    [u,r2] = linsolve(R,tmp,opts);
    num_linsolve = num_linsolve + 1;

    % Check if linsolve is singular; recompute QR decomposition if so.
    if(r2 < tol)
        disp("Warning: Linear system is nearly singular." + ...
            " Recomputing QR decomposition...")
        [Q,R] = qr(A(:,active_set));
        tmp = (b.'*Q).';
        u = R\tmp;
    end    
end

% Return the solution and its residual
x = zeros(n,1); x(active_set) = u;
d = -theta;
ind = find(eqset);
eqset(ind(~active_set)) = 0;
end