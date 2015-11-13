function idx = spectral(W, k)
%SPECTRUAL spectral clustering
%   Input:
%     W: Adjacency matrix, N-by-N matrix
%     k: number of clusters
%   Output:
%     idx: data point cluster labels, n-by-1 vector.

% YOUR CODE HERE
L = diag(sum(W, 2)) - W;
[vec, e] = eig(L);
e = diag(e);
[~, I] = sort(e);
idx = litekmeans(vec(:, I(1:k)), k);

end
