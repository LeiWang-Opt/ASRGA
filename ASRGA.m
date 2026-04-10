%--------------------------------------------------------------------------
function [x, out] = ASRGA(problem, manifold, xinit, p, opts)
%--------------------------------------------------------------------------

if nargin < 4
    error('at least four inputs: [x, out] = ASRGA(problem, manifold, xinit, p)');
elseif nargin < 5
    opts = [];
end

% size of the problem
[n, s] = size(xinit);

%--------------------------------------------------------------------------
% options for the solver

if ~isfield(opts,     'eta');     opts.eta     = 1e-6;  end
if ~isfield(opts,   'delta');     opts.delta   = 1e-1;  end
if ~isfield(opts,   'gamma');     opts.gamma   = 1e-1;  end
if ~isfield(opts,     'tol');     opts.tol     = 1e-3;  end
if ~isfield(opts, 'maxiter');     opts.maxiter =  1e3;  end

if isfield(opts, 'error')
    compute_error = opts.error;
    error_flag = 1;
else
    error_flag = 0;
end

if isfield(opts, 'maxtime') && isnumeric(opts.maxtime) && ~isempty(opts.maxtime)
    maxtime = opts.maxtime;
    time_flag = 1;
    start = tic;
else
    time_flag = 0;
end

%--------------------------------------------------------------------------
% copy parameters

eta     = opts.eta;
delta   = opts.delta;
gamma   = opts.gamma;
tol     = opts.tol;
maxiter = opts.maxiter;

%--------------------------------------------------------------------------
% initial setup

x = xinit;

retr = manifold.retraction;
proj = manifold.projection;

% monotone queue storing candidates
qmax = ceil(maxiter / 2);

q_x         = cell(qmax,  1);  % candidate points
q_rgradnorm = zeros(qmax, 1);  % Riemannian gradient norms
q_mu        = zeros(qmax, 1);  % smoothing parameters
q_index     = zeros(qmax, 1);  % iteration indices

head = 1;
tail = 0;

rgnorm = zeros(maxiter, 1);

if error_flag
    error_hist = zeros(maxiter, 1);
end

if time_flag
    time_hist = zeros(maxiter, 1);
end


%--------------------------------------------------------------------------
% main loop

for iter = 1 : 1: maxiter

    %----- compute smoothing parameter -----%
    mu = delta * iter^(-1 / (4 - p));
    %mu = mu / 1.01;

    [~, grad] = problem(x, mu);
    rgrad = proj(x, grad);
    rgrad_norm = norm(rgrad, 'fro');

    %----- update monotone queue -----%
    while tail >= head && q_rgradnorm(tail) >= rgrad_norm
        tail = tail - 1;
    end

    tail = tail + 1;
    q_x{tail}         = x;
    q_rgradnorm(tail) = rgrad_norm;
    q_mu(tail)        = mu;
    q_index(tail)     = iter;


    %----- enforce sliding window -----%
    L = ceil(iter / 2);
    while q_index(head) < L
        head = head + 1;
    end

    %----- choose candidate -----%
    x_best         = q_x{head};
    mu_best        = q_mu(head);
    rgradnorm_best = q_rgradnorm(head);

    %info_rgradnorm(iter) = rgrad_norm;
    rgnorm(iter) = mu_best^((2 - p) / 2) * rgradnorm_best;

    if error_flag
        error_hist(iter) = compute_error(x_best);
    end

    %----- stopping criterion -----%
    %if (rgradnorm_best <= tol) && (mu_best <= tol)
    if rgradnorm_best <= tol && time_flag == 0
        break;
    end

    if time_flag == 1
        now = toc(start);
        time_hist(iter) = now;
        if now >= maxtime
            break;
        end
    end

    %eta = sqrt(eta.^2 + (mu / delta)^(2 - p) * abs(rgrad).^2);
    %x = retr(x, - gamma * (mu / delta)^(2 - p) * rgrad ./ eta);

    eta = sqrt(eta^2 + (mu / delta)^(2 - p) * rgrad_norm^2);
    x = retr(x, - gamma * (mu / delta)^(2 - p) * rgrad / eta);

end


%--------------------------------------------------------------------------
% store the iter. info.

x = x_best;

out.iter = iter;
out.mu_best = mu_best;
out.mu = mu;
out.rgnorm = rgnorm(1 : iter);
out.eta = eta;

if error_flag
    out.error = error_hist(1 : iter);
end

if time_flag
    if time_hist(iter) > maxtime
        time_hist(iter) = maxtime;
    end
    out.time = time_hist(1 : iter);
end

end
%--------------------------------------------------------------------------


