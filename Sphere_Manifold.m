function S = Sphere_Manifold()

S.retraction = @(x, v) retr(x, v);
S.projection = @(x, v) proj_tangent(x, v);

end


function y = retr(x, v)

y = x + v;
y = y / norm(y, 'fro');

end


function d = proj_tangent(x, v)

d = v - x * (x' * v);

end