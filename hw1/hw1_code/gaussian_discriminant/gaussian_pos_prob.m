function p = gaussian_pos_prob(X, Mu, Sigma, Phi)
%GAUSSIAN_POS_PROB Posterior probability of GDA.
%   p = GAUSSIAN_POS_PROB(X, Mu, Sigma) compute the posterior probability
%   of given N data points X using Gaussian Discriminant Analysis where the
%   K gaussian distributions are specified by Mu, Sigma and Phi.
%
%   Inputs:
%       'X'     - M-by-N matrix, N data points of dimension M.
%       'Mu'    - M-by-K matrix, mean of K Gaussian distributions.
%       'Sigma' - M-by-M-by-K matrix (yes, a 3D matrix), variance matrix of
%                   K Gaussian distributions.
%       'Phi'   - 1-by-K matrix, prior of K Gaussian distributions.
%
%   Outputs:
%       'p'     - N-by-K matrix, posterior probability of N data points
%                   with in K Gaussian distributions.

[M, N] = size(X);
K = length(Phi);
p = zeros(N, K);
Pd = zeros(N, K);
Px = zeros(N, 1);

coef = zeros(K, 1);
invSigma = zeros(size(Sigma));
for i = 1:K
    coef(i) = 1 / (2 * pi)^(M / 2) * sqrt(det(Sigma(:, :, i)));
    invSigma(:, :, i) = inv(Sigma(:, :, i));
end

for i = 1:N
    for j = 1:K
        Pd(i, j) = coef(j) * exp(-0.5 * (X(:, i) - Mu(:, j))' * invSigma(:, :, j) * (X(:, i) - Mu(:, j)));
    end
end

for i = 1:N
    Px(i) = sum(Pd(i, :) .* Phi);
end

for i = 1:N
    p(i, :) = Pd(i, :) .* Phi / Px(i);
%     for j = 1:K
%         p(i, j) = Pd(i, j) * Phi(j) / Px(i);
%     end
end

% Your code HERE
