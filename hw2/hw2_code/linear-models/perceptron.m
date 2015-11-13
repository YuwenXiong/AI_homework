function [w, iter] = perceptron(X, y)
%PERCEPTRON Perceptron Learning Algorithm.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           iter: number of iterations
%

% YOUR CODE HERE
[P, N] = size(X);
w = rand(P + 1, 1);
maxiter = 2000;
iter = 0;
alpha = 0.1;
X = [ones(1, N); X];
while iter < maxiter
    for i = 1:N
        w = w + alpha * (y(i) - sign(w' * X(:, i))) * X(:, i);
    end
    iter = iter + 1;
    if sum(sign(w' * X) ~= y) == 0
        break
    end
end

end
