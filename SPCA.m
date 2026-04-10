function [fval, grad] = SPCA(x, A, lambda, mu, type)

if nargin < 4
    mu = -1;
    type = 1;
end

if nargin < 5
    type = 1;
end

Ax = A * x;
fval = - 0.5 * sum(dot(Ax, Ax));

if type == 1

    ax = abs(x);

    idx1 = (ax > mu);
    idx2 = ~idx1;

    s = zeros(size(x));
    s(idx1) = ax(idx1);
    s(idx2) = x(idx2).^2 ./ (2 * mu) + mu / 2;

    fval = fval + lambda * sum(s(:));

    if nargout > 1
        ds = zeros(size(x));
        ds(idx1) = sign(x(idx1));
        ds(idx2) = x(idx2) ./ mu;
        grad = - A.' * Ax + lambda * ds;
    end

end

if type == 2
    % smoothing by Moreau envelope

    x_prox = prox_l1(x, mu);
    x_diff = x - x_prox;
    fval = fval + lambda * ...
        ( abs(x_prox) + norm(x_diff, 'fro')^2 / (2 * mu) );

    if nargout > 1
        grad = - A.' * Ax + lambda * x_diff / mu;
    end

end

end


function x_prox = prox_l1(x, lambda)

a = abs(x) - lambda;
act_set = (a > 0);
x_prox = (act_set .* sign(x)) .* a;

end