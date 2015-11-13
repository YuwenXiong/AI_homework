function [w, num] = svm(X, y)
%SVM Support vector machine.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           num:  number of support vectors
%

% YOUR CODE HERE
[P, N] = size(X);
H = ones(P + 1, P + 1);
H(1, :) = 0;
H(:, 1) = 0;
a = zeros(N, P + 1);
for i = 1:N
    a(i, :) = [y(i); y(i) * X(:, i)]';
end
[w,~,~,~,LAMBDA] = quadprog(H, zeros(P + 1, 1), -a, -ones(N, 1), [], [], [], [], [], optimset('Display', 'off'));
% num = sum(LAMBDA.ineqlin > 1e-3);
num = sum(abs(y .* (w' * [ones(1, size(X, 2)); X]) - 1) < 1e-3);
end
