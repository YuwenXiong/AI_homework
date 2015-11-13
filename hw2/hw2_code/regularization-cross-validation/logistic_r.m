function w = logistic_r(X, y, lambda)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%           lambda: regularization parameter.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE

[P, N] = size(X);
X = [ones(1, N); X];

w = rand(P + 1, 1);
lr = 0.001;

for i = 1:100000
    w = w + lr * (X * (y - sigmoid(w' * X))' - lambda * [0; w(2:end)]);
    if mod(i, 100) == 0
        lr = lr * 0.99;
        if sum((lr * (X * (y - sigmoid(w' * X))' - lambda * [0; w(2:end)])).^2) < 1e-8
            break
        end
    end
end
end
