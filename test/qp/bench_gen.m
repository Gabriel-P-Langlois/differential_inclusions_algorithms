function [X, y, w_star] = bench_gen(n, df, s, sv_target, k_factors, corr_noise)
if nargin < 5, k_factors = []; end
if nargin < 6 || isempty(corr_noise), corr_noise = 0.1; end
w_star = randn(df, 1); w_star = w_star / norm(w_star);
rows = repelem((1:n)', s); cols = zeros(n * s, 1);
for i = 1:n, cols((i-1)*s + (1:s)) = randperm(df, s); end
if isempty(k_factors)
    vals = randn(n * s, 1);
else
    L = randn(df, k_factors); F = randn(n, k_factors); vals = zeros(n * s, 1);
    for i = 1:n
        idx = (i-1)*s + (1:s); c = cols(idx);
        vals(idx) = L(c, :) * F(i, :)' + corr_noise * randn(s, 1);
    end
end
X = sparse(rows, cols, vals, n, df);
rn = sqrt(sum(X.^2, 2)); rn(rn == 0) = 1; X = X ./ rn;
scores = X * w_star; y = sign(scores); y(y == 0) = 1;
[~, ord] = sort(abs(scores), 'ascend'); nflip = round(sv_target * n);
y(ord(1:nflip)) = -y(ord(1:nflip));
end
