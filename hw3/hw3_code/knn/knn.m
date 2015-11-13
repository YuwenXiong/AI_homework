function y = knn(X, X_train, y_train, K)
%KNN k-Nearest Neighbors Algorithm.
%
%   INPUT:  X:         testing sample features, P-by-N_test matrix.
%           X_train:   training sample features, P-by-N matrix.
%           y_train:   training sample labels, 1-by-N row vector.
%           K:         the k in k-Nearest Neighbors
%
%   OUTPUT: y    : predicted labels, 1-by-N_test row vector.
%

% YOUR CODE HERE
y = zeros(1, size(X, 2));
D = EuDist2(X', X_train');
for i = 1:size(X, 2)
%     D = sum(bsxfun(@minus, X_train, X(:, i)).^2, 1);
    [~, I] = sort(D(i, :));
    nn = y_train(I(1:K));
    if length(unique(nn)) == 1
        y(i) = unique(nn);
    else
        [a, b] = hist(nn, unique(nn));
        [~, I] = max(a);
        y(i) = b(I(1));
    end
end

