function F = dg_project(S, fun)
%DG_PROJECT  L2 projection of fun(x, y) onto the DG space of dg_setup,
%   integrals by the tensor (k+1)-point Gauss rule (LHTZ section 3.1).
%   Returns F, nb-by-Nc.  With the orthonormal basis,
%   F(r, c) = (1/|E|) int_E fun*phi_r = sum_q wq * fun(q) * phi_r(q).
    F = S.Phi.' * (S.wq .* fun(S.Xq, S.Yq));
end
