function [A, C] = dg_assemble(S, Dfun, bfun, sigma, h)
%DG_ASSEMBLE  NIPG diffusion matrix A and Lax-Friedrichs convection matrix
%   C of Liu-Hu-Taitano-Zhang section 3.1, for time-independent D and b:
%
%       (A*f)_r = a_diff(f, phi_r),     (C*f)_r = a_conv(f, phi_r),
%
%   with f and phi_r expanded in the basis of dg_setup (dof index
%   (c-1)*nb + r).  The scheme (21) is then
%       dx^2 f^n + tau*A f^n = dx^2 f^{n-1} + tau*C f^{n-1}
%   (eps^-1 and m/T absorbed into A and C by the caller's D and b).
%
%   a_diff: volume term, then on INTERIOR faces only (Gamma_h) the NIPG
%   terms  - {D grad f . n}[chi] + {D grad chi . n}[f] + (sigma/h)[f][chi].
%   a_conv: volume term  int f b . grad chi, minus the Lax-Friedrichs flux
%   (19b) on interior faces, alpha_e = max over the face points of |b.n_e|.
%   The convective flux on boundary faces is zero: section 3.1 states
%   a_conv(f, 1) = 0, which requires it.
%
%   Dfun(x,y) returns [D11, D12, D22]; bfun(x,y) returns [b1, b2]; both
%   evaluated elementwise at the quadrature points.

    nb = S.nb;  Nc = S.Nc;  dx = S.dx;  Nd = nb*Nc;
    [aa, bb] = ndgrid(1:nb, 1:nb);
    aa = aa(:);  bb = bb(:);
    IA = {};  JA = {};  VA = {};  VC = {};

    % Volume terms (no dx factor for diffusion; dx for convection).
    wq = S.wq;
    [D11, D12, D22] = Dfun(S.Xq, S.Yq);
    [B1, B2] = bfun(S.Xq, S.Yq);
    Av = kr(S.dPs, S.dPs, wq)*D11 + (kr(S.dPs, S.dPt, wq) + kr(S.dPt, S.dPs, wq))*D12 ...
       + kr(S.dPt, S.dPt, wq)*D22;
    Cv = dx*(kr(S.dPs, S.Phi, wq)*B1 + kr(S.dPt, S.Phi, wq)*B2);
    cells = (1:Nc);
    IA{end+1} = reshape(aa + (cells - 1)*nb, [], 1);
    JA{end+1} = reshape(bb + (cells - 1)*nb, [], 1);
    VA{end+1} = Av(:);  VC{end+1} = Cv(:);

    % Interior faces.  Side 1 = cell of lower index (normal points out of
    % it), side 2 = the other; jump [chi] = chi|1 - chi|2.
    w1 = S.w1;
    sgn = [1, -1];
    for dir = 1:2
        if dir == 1                       % vertical faces, n = (1, 0)
            Xf = S.vX;  Yf = S.vY;  cm = S.vm;  cp = S.vp;
            trs = {S.tr.E, S.tr.W};
        else                              % horizontal faces, n = (0, 1)
            Xf = S.hX;  Yf = S.hY;  cm = S.hm;  cp = S.hp;
            trs = {S.tr.N, S.tr.S};
        end
        if isempty(cm), continue, end
        [F11, F12, F22] = Dfun(Xf, Yf);
        [b1, b2] = bfun(Xf, Yf);
        if dir == 1
            Dn1 = F11;  Dn2 = F12;  bn = b1;
        else
            Dn1 = F12;  Dn2 = F22;  bn = b2;
        end
        alpha = max(abs(bn), [], 1);
        Nf = numel(cm);
        cl = {cm, cp};
        for r = 1:2
            for c = 1:2
                Ta = trs{r};  Tb = trs{c};  sr = sgn(r);  sc = sgn(c);
                Ablk = -0.5*sr*(kr(Ta.P, Tb.dS, w1)*Dn1 + kr(Ta.P, Tb.dT, w1)*Dn2) ...
                       + 0.5*sc*(kr(Ta.dS, Tb.P, w1)*Dn1 + kr(Ta.dT, Tb.P, w1)*Dn2) ...
                       + (sigma*dx/h)*sr*sc*(kr(Ta.P, Tb.P, w1)*ones(numel(w1), Nf));
                Cblk = -sr*dx*(kr(Ta.P, Tb.P, w1)*(0.5*bn + 0.5*sc*alpha));
                IA{end+1} = reshape(aa + (cl{r}.' - 1)*nb, [], 1); %#ok<AGROW>
                JA{end+1} = reshape(bb + (cl{c}.' - 1)*nb, [], 1); %#ok<AGROW>
                VA{end+1} = Ablk(:);  VC{end+1} = Cblk(:);         %#ok<AGROW>
            end
        end
    end

    I = vertcat(IA{:});  J = vertcat(JA{:});
    A = sparse(I, J, vertcat(VA{:}), Nd, Nd);
    C = sparse(I, J, vertcat(VC{:}), Nd, Nd);
end


function K = kr(Pa, Pb, w)
%KR  nb^2-by-np matrix whose column p is w(p)*vec(Pa(p,:)' * Pb(p,:)), so
%   that K*v gives, per face or cell, the block sum_p w_p v_p Pa(p,a) Pb(p,b).
    [np, nb] = size(Pa);
    K = zeros(nb*nb, np);
    for p = 1:np
        K(:, p) = w(p) * reshape(Pa(p,:).' * Pb(p,:), [], 1);
    end
end
