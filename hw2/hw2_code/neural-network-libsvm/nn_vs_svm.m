load('digit_data', 'X', 'y');
load('weights', 'Theta1', 'Theta2');

p = feedforward(Theta1, Theta2, X);
fprintf('Error rate for NN is %f.\n', length(find(p ~= y))/length(p));

train_X = X(:, 1:2500);
train_y = y(1:2500);
test_X = X(:, 2501:end);
test_y = y(2501:end);

% YOUR CODE HERE
% Trainning and testing using one-vs-all with LIBLINEAR

num_labels = 10;
model = cell(num_labels, 1);
for i = 1:num_labels
    model{i} = train(double(train_y == i)', sparse([ones(1, size(train_X, 2)); train_X]'), '-q col');
end

test_label = zeros(num_labels, size(test_y, 2));
for i = 1:num_labels
    [predicted_label, accuracy, decision_values] = predict(test_y', sparse([ones(1, size(test_X, 2)); test_X]'), model{i}, '-q col');
    test_label(i, :) = (model{i}.Label(2) * 2 - 1) * decision_values;
%     test_label(i, :) = model{i}.w * test_X;
end
[V, I] = sort(test_label, 1, 'ascend');
p = I(1, :);
fprintf('Error rate for SVM is %f.\n', length(find(p ~= test_y))/length(p));
