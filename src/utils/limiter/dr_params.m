function [c, lam, th] = dr_params(u, m, M)
%DR_PARAMS  Parameters c and lambda of Algorithm DR, Remark 2 of Liu-Hu-
%   Taitano-Zhang (CAMWA 192, 2025), which is rule (1.9) of Liu-Riviere-
%   Shen-Zhang (SISC 46(3), 2024): theta = acos(sqrt(rhat/N)), with rhat the
%   number of entries of u outside [m, M].
    th = acos(sqrt(nnz(u < m | u > M) / numel(u)));
    if th > 3*pi/8
        c = 1/2;  lam = 4/(2 - cos(2*th));
    elseif th > pi/4
        c = 1/(cos(th) + sin(th))^2;
        lam = 2/(1 + 1/(1 + cot(th)) - c);
    else
        c = 1/(cos(th) + sin(th))^2;  lam = 2;
    end
end
