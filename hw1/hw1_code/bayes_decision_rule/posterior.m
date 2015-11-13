function p = posterior(x)
%POSTERIOR Two Class Posterior Using Bayes Formula
%
%   INPUT:  x, features of different class, C-By-N vector
%           C is the number of classes, N is the number of different feature
%
%   OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
%

[C, N] = size(x);
l = likelihood(x);
total = sum(sum(x));
%TODO
p_omega = sum(x, 2) / total;
p_x = zeros(C, N);
for i = 1:N
    p_x(:, i) = l(:, i) .* p_omega;
end
p_x = sum(p_x, 1);

p = zeros(C, N);
for i = 1:N
    p(:, i) = l(:, i) .* p_omega / p_x(i);
end
