function w = logistic(X, y)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE

[P, N] = size(X);
X = [ones(1, N); X];
% X(:) = (X(:) - mean(X(:))) / std(X(:));
w = rand(P + 1, 1);
lr = 0.2;
for i = 1:10000
    w = w + lr * (X * (y - sigmoid(w' * X))');
    if mod(i, 100) == 0
        lr = lr * 0.9;
        if sum((lr * (X * (y - sigmoid(w' * X))')).^2) < 1e-8
            break
        end
    end
end


end
