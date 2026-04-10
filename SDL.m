function [fval, grad] = SDL(X, Y, p, mu)
% smoothing objective and Euclidean gradient of sparse dictionary learning
% X: n x n orthogonal dictionary
% Y: n x m data matrix (each column Yi)
% p: 0 < p < 1
% mu: smoothing parameter

if nargin < 4
    mu = -1;
end

[~, s] = size(X);
[n, m] = size(Y);

% Z = Y^T * X  (m x n)
Z = Y' * X;

% smoothing phi_mu(t)
absZ = abs(Z);
phi = zeros(m, s);
phi(absZ > mu) = absZ(absZ > mu);
phi(absZ <= mu) = Z(absZ <= mu) .^ 2 / (2 * mu) + mu / 2;

% objective value
fval = (1 / m) * sum(sum(phi .^ p));

if nargout > 1
    % derivative phi_mu'(t)
    phi_prime = zeros(m, s);
    phi_prime(absZ > mu) = sign(Z(absZ > mu));
    phi_prime(absZ <= mu) = Z(absZ <= mu) / mu;
    % gradient factor
    W = (phi .^ (p - 1)) .* phi_prime;
    % Euclidean gradient
    grad = (p / m) * Y * W;
end

end