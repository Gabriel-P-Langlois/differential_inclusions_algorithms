function prox = l1_prox_operator(x,t)
%   L1_PROX_OPERATOR    Compute the soft thresholding operator
%                       of x with parameter t
%
% -------------------------------------------------------------------------
%   INPUTS
%       x       -   n-dimensional row or column vector
%       t       -   positive scalar
%
%   OUTPUTS
%       prox    -   n-dimensional row or column vector
%                   optimal solution of the problem
%                   argmin_{y} 0.5*||x-y||_{2}^{2} + ||y||_{1}
%
% -------------------------------------------------------------------------

prox = abs(x)-t;
prox = sign(x).*(prox+abs(prox))*0.5;
end