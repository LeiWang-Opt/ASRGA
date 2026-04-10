%--------------------------------------------------------------------------
function [x, out] = RSSD(problem, manifold, xinit, opts)
% Riemannian Smoothing Steepest Descent
%
% Inputs:
%   x0          : initial point on manifold M
%   f_tilde     : handle, f_tilde(x, mu) -> scalar \tilde f(x, mu)
%   grad_tilde  : handle, grad_tilde(x, mu) -> Euclidean gradient in ambient space
%   mani        : struct with
%                 mani.proj(x, g)  -> projection of ambient vector g onto T_x M
%                 mani.retr(x, xi) -> retraction R_x(xi), xi in T_x M
%   opts        : parameters
%       opts.deltaOpt (>=0), opts.delta0 (>0)
%       opts.muOpt    (>=0), opts.mu0    (>0)
%       opts.sigma in (0,1), opts.beta in (0,1), opts.alphaBar > 0
%       opts.thetaDelta in (0,1), opts.thetaMu in (0,1)
%       opts.maxIter (default 10000), opts.maxLS (default 50)
%       opts.verbose (default true)
%
% Outputs:
%   x     : final iterate
%   out   : struct with history

% options for the solver

if ~isfield(opts,   'alpha');     opts.alpha   = 1;     end
if ~isfield(opts,    'beta');     opts.beta    = 0.5;   end
if ~isfield(opts,   'delta');     opts.delta   = 1e-1;  end
if ~isfield(opts,      'mu');     opts.mu      = 1e-1;  end
if ~isfield(opts,   'sigma');     opts.sigma   = 1e-1;  end
if ~isfield(opts,   'theta');     opts.theta   = 0.5;   end
if ~isfield(opts,     'tol');     opts.tol     = 1e-3;  end
if ~isfield(opts, 'maxiter');     opts.maxiter =  1e3;  end
if ~isfield(opts,  'record');     opts.record  =    0;  end

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

% size of the problem
[n, s] = size(xinit);

% copy parameters
alpha   = opts.alpha;
beta    = opts.beta;
delta   = opts.delta;
mu      = opts.mu;
sigma   = opts.sigma;
theta   = opts.theta;
tol     = opts.tol;
maxiter = opts.maxiter;
record  = opts.record;


% initial setup
x = xinit;

retr = manifold.retraction;
proj = manifold.projection;

rgnorm = zeros(maxiter, 1);

if error_flag
    error_hist = zeros(maxiter, 1);
end

if time_flag
    time_hist = zeros(maxiter, 1);
end

maxls = 1000;
t = alpha;

for iter = 1: 1 : maxiter

    % compute Riemannian gradient
    [f0, gE] = problem(x, mu);
    gR = proj(x, gE);

    eta = -gR;
    gnorm = norm(gR, 'fro');

    rgnorm(iter) = gnorm;

    if error_flag
        error_hist(iter) = compute_error(x);
    end

    if record && (mod(iter, 50) == 0)
        fprintf('[%5d] f=%.6e, ||grad||=%.3e, mu=%.3e, delta=%.3e\n', ...
            iter, f0, gnorm, mu, delta);
    end

    % stopping check
    if time_flag == 1
        now = toc(start);
        time_hist(iter) = now;
        if now >= maxtime
            break;
        end
    end

    %if (gnorm <= tol) && (mu <= tol)
    if gnorm <= tol && time_flag == 0
        break;
    elseif gnorm <= delta
        % shrink smoothing and tolerance, keep x
        mu = theta * mu;
        delta = theta * delta;
        continue;
    else
        % Armijo backtracking on the manifold
        gradnorm2 = gnorm^2;
        %t = alpha;
        m = 0;

        while true
            xtrial = retr(x, t * eta);
            ftrial = problem(xtrial, mu);

            % Armijo condition: f(Rx(t*eta)) <= f(x) - sigma * t * ||grad||^2
            if ftrial <= f0 - sigma * t * gradnorm2
                break;
            end

            m = m + 1;
            if m > maxls
                %warning('Line search exceeded maxls=%d. Accepting current trial step.', maxls);
                break;
            end
            t = beta^m * alpha;
        end

        x = xtrial;
    end

end


% trim history
out.iter = iter;
out.mu = mu;
out.rgnorm = rgnorm(1 : iter);

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
