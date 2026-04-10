function [fval, grad] = DPCP(x, E, p, mu)

if nargin < 4
    mu = -1;
end

y = E * x;
ay = abs(y);

idx1 = (ay > mu);
idx2 = ~idx1;

s = zeros(size(y));
s(idx1) = ay(idx1);
s(idx2) = y(idx2).^2 ./ (2 * mu) + mu / 2;

fval = sum(s.^p);

if nargout > 1
    ds = zeros(size(y));
    ds(idx1) = sign(y(idx1));
    ds(idx2) = y(idx2) ./ mu;
    z = p * s.^(p-1) .* ds;
    grad = E.' * z;
end

end