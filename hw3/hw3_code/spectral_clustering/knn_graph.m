function W = knn_graph(X, k, threshold)
%KNN_GRAPH Construct W using KNN graph
%   Input: X - data point features, n-by-p maxtirx.
%          k - number of nn.
%          threshold - distance threshold.
%
%   Output:W - adjacency matrix, n-by-n matrix.

% YOUR CODE HERE
t = 1;
N = size(X, 1);
D = EuDist2(X, X);
W = zeros(N, N);
D = D + eye(N) * max(D(:)) * 10;
[V, I] = sort(D, 2);
for i = 1:N;
    W(i, I(i, V(i, 1:k) < threshold)) = 1;
%     W(i, I(i, V(i, 1:k) < threshold)) = ...
%     exp(-sqrt(sum(bsxfun(@minus, X(I(i, V(i, 1:k) < threshold), :), X(i, :)).^2, 2)) / 2*t^2);
end
end
