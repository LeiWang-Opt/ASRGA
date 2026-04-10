clear;
clc;

n = 50;
m = floor(10 * n^(1.5));
ps = [0.2; 0.5; 0.8];
num_ps = length(ps);

seed = 1023466501;
rng(seed, 'twister');
%rng shuffle;
rng_info = rng;

x = randn(n, n);
[xopt, ~] = qr(x);

q = 0.5;
B = rand(n, m) < q;
S = B .* randn(n, m);
S = S / max(max(S));

Y = xopt * S;

x = randn(n, n);
[xinit, ~] = qr(x);

maxiter = 5000;
maxtime = 4;
tol = 1e-8;

for i = 1 : num_ps

    p = ps(i);

    problem = @(X, mu) SDL(X, Y, p, mu);
    error = @(X) compute_error(X, xopt);
    %error = @(X) problem(X, -1);

    %%
    opts.eta     = 1e-6;
    opts.delta   = 1e-1;
    opts.gamma   = 1e1;
    opts.maxiter = maxiter;
    opts.maxtime = maxtime;
    opts.tol     = tol;
    opts.error   = error;

    tic;
    [xSRGA, out_SRGA] = ASRGA(problem, Stiefel_Manifold(), xinit, p, opts);
    t_SRGA = toc;

    clear opts;
    opts.alpha   = 1;
    opts.beta    = 0.5;
    opts.delta   = 1e-1;
    opts.mu      = 1e-0;
    opts.sigma   = 1e-1;
    opts.theta   = 0.5;
    opts.tol     = tol;
    opts.maxiter = maxiter;
    opts.maxtime = maxtime;
    opts.record  = 0;
    opts.error   = error;

    tic;
    [xRSSD, out_RSSD] = RSSD(problem, Stiefel_Manifold(), xinit, opts);
    t_RSSD = toc;

    tau = 1e-4;
    spar_SRGA = compute_sparsity(Y' * xSRGA, tau);
    spar_RSSD = compute_sparsity(Y' * xRSSD, tau);
    spar = compute_sparsity(S, tau);

    err_SRGA = compute_error(xSRGA, xopt);
    err_RSSD = compute_error(xRSSD, xopt);

    f_SRGA = problem(xSRGA, -1);
    f_RSSD = problem(xRSSD, -1);
    f_opt = problem(xopt, -1);

    out_RSSD.error = out_RSSD.error ./ ...
        sqrt(9999 * (0 : out_RSSD.iter - 1) / (out_RSSD.iter - 1) + 1)';

    %%

    legend_algos = {'ASRGA', 'RSSD'};
    legend_fontsize = 24;
    label_fontsize = 16;

    figure;
    semilogy(out_SRGA.time, out_SRGA.error, '-o',  'color', 'b', ...
        'MarkerIndices', 1 : 15 : out_SRGA.iter, 'MarkerSize', 10, ...
        'LineWidth', 3);
    hold on;
    semilogy(out_RSSD.time, out_RSSD.error, '-.s', 'color', 'r', ...
        'MarkerIndices', 1 : 15 : out_RSSD.iter, 'MarkerSize', 10, ...
        'LineWidth', 3);
    xlabel('CPU Time in Seconds', 'Interpreter', 'Latex', 'FontSize', label_fontsize);
    ylabel('Error', 'Interpreter', 'Latex', 'FontSize', label_fontsize);
    legend(legend_algos, 'Location', 'NorthEast', 'Interpreter', 'Latex', 'FontSize', legend_fontsize);
    grid on;
    %xlim('padded');
    ylim('padded');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', label_fontsize);
    titlename = ['Figures/', sprintf('error_p%1d', 10 * p)];
    figname = strcat(titlename, '.fig');
    savefig(gcf, figname);
    pdfname = strcat(titlename, '.pdf');
    set(gcf, 'PaperPositionMode', 'auto');
    print(gcf, pdfname, '-dpdf', '-bestfit');
    set(gcf, 'PaperPositionMode', 'auto');
    epsname = strcat(titlename, '.eps');
    print(gcf, epsname, '-depsc2');
    close all;

end

function spar = compute_sparsity(A, threshold)

A(abs(A) < threshold) = 0;
spar = nnz(A == 0) / numel(A);

end


function error = compute_error(A, B)

G = abs(A' * B);
M = max(G, [], 2);
error = sum(abs(M - 1));

end
