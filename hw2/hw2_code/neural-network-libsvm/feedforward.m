function p = feedforward(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%
%   Input:  Theta1 -- weights between input-hidden layers, 401x25 matrix
%           Theta2 -- weights between hidden-output layers, 26x10 matrix
%                X -- test set, 400xP matrix, P is size of testing set
%
%   Output: p -- predicted labels, 1xP row vector

% Note:
% The matrix X contains the examples in columns.
% The matrices Theta1 and Theta2 contain the parameters for each unit in
% column. Specifically, the first column of Theta1 corresponds to the first
% hidden unit in the second layer.

% YOUR CODE HERE
p = sigmoid(Theta1' * [ones(1, size(X, 2)); X]);
p = sigmoid(Theta2' * [ones(1, size(p, 2)); p]);
[~, I] = sort(p, 1, 'descend');
p = I(1, :);
end

function g = sigmoid(z)
g = 1./(1+exp(-z));
end