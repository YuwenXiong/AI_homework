function [idx, ctrs, iter_ctrs] = kmeans(X, K)
%KMEANS K-Means clustering algorithm
%
%   Input: X - data point features, n-by-p maxtirx.
%          K - the number of clusters
%
%   OUTPUT: idx  - cluster label
%           ctrs - cluster centers, K-by-p matrix.
%           iter_ctrs - cluster centers of each iteration, K-by-p-by-iter
%                       3D matrix.

% YOUR CODE HERE

N = size(X, 1);
ctrs = X(randsample(N, K), :);
idx = zeros(N, 1);
for iter = 1:1000
    preidx = idx;
    D = EuDist2(X, ctrs);
    [~, idx] = min(D, [], 2);
    for i = 1:K
        ctrs(i, :) = mean(X(idx == i, :));
    end
    iter_ctrs(:, :, iter) = ctrs;
    if (idx == preidx)
        fprintf('end with iter: %d\n', iter);
        break
    end
end
end
