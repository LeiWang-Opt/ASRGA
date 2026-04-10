clear;
clc;

%  data seq  frame
% 09_26_005: 1 45 120 137 153
% 09_26_048: 0 21
% 09_29_071: 221 328 441 881

date  = 26;
seq   = 5;
frame = 153;

data_path = './KITTI/';
data = [data_path, sprintf('09_%02d_%03d/%02d_%03d_%04d_data.mat', ...
    date, seq, date, seq, frame)];

S = load(data);

%----- inliers -----%
A = S.inliers;
inliers = A ./ vecnorm(A, 2, 1);

%----- outliers -----%
A = S.outliers;
outliers = A ./ vecnorm(A, 2, 1);

A = [inliers, outliers];
% idx = randperm(size(A,2));
% A = A(:, idx);

n_points     = length(A);
n_inliers    = size(inliers, 2);
n_outliers   = size(outliers, 2);
inlier_ratio = n_inliers / n_points;

[x_opt, ~]  = eigs(inliers * inliers.', 1, 'sa');
[x_init, ~] = eigs(A * A.', 1, 'sa');

[E, R] = qr(A.', 0);

[m, n] = size(A.');
p = 0.5;

problem = @(x, mu) DPCP(x, E, p, mu);

maxiter = 300;
tol = 1e-10;
maxtime = 0.5;

opts.eta     = 1e-6;
opts.delta   = 1e-3;
opts.gamma   = 1e-0;
opts.maxiter = maxiter;
opts.maxtime = maxtime;
opts.tol     = tol;

tic;
[xSRGA, out_SRGA] = ASRGA(problem, Sphere_Manifold(), x_init, p, opts);
t_SRGA = toc;

xSRGA = R \ xSRGA;
xSRGA = xSRGA / norm(xSRGA, 2);
err_SRGA = sqrt(1 - (x_opt.' * xSRGA)^2);


clear opts;
opts.alpha   = 1;
opts.beta    = 0.5;
opts.delta   = 1e-1;
opts.mu      = 5e-1;
opts.sigma   = 1e-1;
opts.theta   = 0.5;
opts.tol     = tol;
opts.maxiter = maxiter;
opts.maxtime = maxtime;
opts.record  = 0;

tic;
[xRSSD, out_RSSD] = RSSD(problem, Sphere_Manifold(), x_init, opts);
t_RSSD = toc;


xRSSD = R \ xRSSD;
xRSSD = xRSSD / norm(xRSSD, 2);
err_RSSD = sqrt(1 - (x_opt.' * xRSSD)^2);


%% visualization

results = struct();
results.SRGA.B = xSRGA;
results.RSSD.B = xRSSD;
results.n_inliers = n_inliers;
results.n_points = n_points;
results.n_outliers = n_outliers;
results.inlier_ratio = inlier_ratio;

Plot_Image(date, seq, frame, data_path, results);

titlename = ['Frames/', sprintf('%02d_%03d_%04d', date, seq, frame)];
figname = strcat(titlename, '.fig');
savefig(gcf, figname);
pdfname = strcat(titlename, '.pdf');
set(gcf, 'PaperPositionMode', 'auto');
%print(gcf, pdfname, '-dpdf', '-bestfit');
exportgraphics(gcf, pdfname, 'ContentType', 'vector');
set(gcf, 'PaperPositionMode', 'auto');
epsname = strcat(titlename, '.eps');
%print(gcf, epsname, '-depsc2');
exportgraphics(gcf, epsname, 'ContentType', 'vector');
close all;
