%% Ridge Regression
load('digit_train', 'X', 'y');

% Do feature normalization
X = normalize(X);

% Do LOOCV
lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
E_vals = zeros(length(lambdas), 1);
for i = 1:length(lambdas)
    E_val = 0;
    for j = 1:size(X, 2)
        X_ = X(:, setdiff(1:size(X, 2), j)); y_ = y(setdiff(1:size(X, 2), j)); % take point j out of X
        w = ridge(X_, y_, lambdas(i));
        E_val = E_val + (sign(w' * [1; X(:, j)]) ~= y(j));
    end
    % Update lambda according validation error
    E_vals(i) = E_val;
end
[~, I] = sort(E_vals);
lambda = lambdas(I(1));
fprintf('determine lambda: %f\n', lambda);
w = ridge(X, y, 0);
w_r = ridge(X, y, lambda);
% Compute training error
E_train = sum(sign(w' * [ones(1, size(X, 2)); X]) ~= y) / size(y, 2);
E_train_r = sum(sign(w_r' * [ones(1, size(X, 2)); X]) ~= y) / size(y, 2);

load('digit_test', 'X_test', 'y_test');
% Do feature normalization
X_test = normalize(X_test);
% Compute test error
E_test = sum(sign(w' * [ones(1, size(X_test, 2)); X_test]) ~= y_test) / size(y_test, 2);
E_test_r = sum(sign(w_r' * [ones(1, size(X_test, 2)); X_test]) ~= y_test) / size(y_test, 2);

fprintf('||w|| with regularization: %f\n', sum(w_r.^2));
fprintf('||w|| without regularization: %f\n', sum(w.^2));
fprintf('training/testing error rate with regularization: %f/%f\n', E_train_r, E_test_r);
fprintf('training/testing error rate without regularization: %f/%f\n', E_train, E_test);

%% Logistic
load('digit_train', 'X', 'y');
X = normalize(X);
y(y == -1) = 0;
lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
E_vals = zeros(length(lambdas), 1);
for i = 1:length(lambdas)
    E_val = 0;
    for j = 1:size(X, 2)
        X_ = X(:, setdiff(1:size(X, 2), j)); y_ = y(setdiff(1:size(X, 2), j)); % take point j out of X
        w = logistic_r(X_, y_, lambdas(i));
        E_val = E_val + (double(sigmoid(w' * [1; X(:, j)]) > 0.5) ~= y(j));
    end
    % Update lambda according validation error
    E_vals(i) = E_val;
end
[~, I] = sort(E_vals);
lambda = lambdas(I(1));
fprintf('determine lambda: %f\n', lambda);
w = logistic_r(X, y, 0);
w_r = logistic_r(X, y, lambda);

E_train = sum(double(sigmoid(w' * [ones(1, size(X, 2)); X]) > 0.5) ~= y) / size(y, 2);
E_train_r = sum(double(sigmoid(w_r' * [ones(1, size(X, 2)); X]) > 0.5) ~= y) / size(y, 2);

load('digit_test', 'X_test', 'y_test');
X_test = normalize(X_test);
y_test(y_test == -1) = 0;
w = logistic_r(X, y, 0);
w_r = logistic_r(X, y, lambda);
E_test = sum(double(sigmoid(w' * [ones(1, size(X_test, 2)); X_test]) > 0.5) ~= y_test) / size(y_test, 2);
E_test_r = sum(double(sigmoid(w_r' * [ones(1, size(X_test, 2)); X_test]) > 0.5) ~= y_test) / size(y_test, 2);

fprintf('||w|| with regularization: %f\n', sum(w_r.^2));
fprintf('||w|| without regularization: %f\n', sum(w.^2));
fprintf('training/testing error rate with regularization: %f/%f\n', E_train_r, E_test_r);
fprintf('training/testing error rate without regularization: %f/%f\n', E_train, E_test);

%% SVM with slack variable
