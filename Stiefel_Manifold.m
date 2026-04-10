function S = Stiefel_Manifold()

S.retraction = @(x, v) retr_polar(x, v);
S.projection = @(x, v) proj_tangent(x, v);

end


function y = retr_polar(x, v)

y = x + v;
[u, ~, v] = svd(y, 'econ');
y = u * v';

end


function d = proj_tangent(x, v)

z = x' * v;
z = (z + z') / 2;
d = v - x * z;

end