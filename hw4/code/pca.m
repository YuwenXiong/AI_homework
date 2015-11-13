function [eigvector, eigvalue] = pca(data)
%PCA	Principal Component Analysis
%
%             Input:
%               data       - Data matrix. Each row vector of fea is a data point.
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The sorted eigvalue of PCA eigen-problem.
%

% YOUR CODE HERE
S = cov(data);
[V, eigvalue] = eig(S);
eigvalue = diag(eigvalue);
[eigvalue, I] = sort(eigvalue, 'descend');
eigvector = V(:, I);
end