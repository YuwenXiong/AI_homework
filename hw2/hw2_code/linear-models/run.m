% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%% Part1: Preceptron
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest = 10 * nTrain;
avgIter = 0;
E_train = 0;
E_test = 0;
for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain + nTest);
    [w_g, iter] = perceptron(X(:, 1:nTrain), y(1:nTrain));
    % Compute training, testing error
    E_train = E_train + sum(sign(w_g' * [ones(1, nTrain); X(:, 1:nTrain)]) ~= y(1:nTrain)) / nTrain;
    E_test = E_test + sum(sign(w_g' * [ones(1, nTest); X(:, nTrain+1:end)]) ~= y(nTrain+1:end)) / (nTest);
    avgIter = avgIter + iter;
end

avgIter = avgIter / nRep;
E_train = E_train / nRep;
E_test  = E_test / nRep;
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
fprintf('Average number of iterations is %d.\n', avgIter);
plotdata(X(:, 1:nTrain), y(1:nTrain), w_f, w_g, 'Pecertron');

%% Part2: Preceptron: Non-linearly separable case
nTrain = 100; % number of training data
[X, y, w_f] = mkdata(nTrain, 'noisy');
[w_g, iter] = perceptron(X, y);


%% Part3: Linear Regression
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
nTest = 10 * nTrain;
E_train = 0;
E_test = 0;

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain + nTest);
    w_g = linear_regression(X(:, 1:nTrain), y(1:nTrain));
    % Compute training, testing error
    E_train = E_train + sum(sign(w_g' * [ones(1, nTrain); X(:, 1:nTrain)]) ~= y(1:nTrain)) / nTrain;
    E_test = E_test + sum(sign(w_g' * [ones(1, nTest); X(:, nTrain+1:end)]) ~= y(nTrain+1:end)) / (nTest);
end

E_train = E_train / nRep;
E_test  = E_test / nRep;
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Linear Regression');

%% Part4: Linear Regression: noisy
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
nTest = 10 * nTrain;
E_train = 0;
E_test = 0;

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain + nTest, 'noisy');
    w_g = linear_regression(X(:, 1:nTrain), y(1:nTrain));
    % Compute training, testing error
    E_train = E_train + sum(sign(w_g' * [ones(1, nTrain); X(:, 1:nTrain)]) ~= y(1:nTrain)) / nTrain;
    E_test = E_test + sum(sign(w_g' * [ones(1, nTest); X(:, nTrain+1:end)]) ~= y(nTrain+1:end)) / (nTest);
end

E_train = E_train / nRep;
E_test  = E_test / nRep;
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Linear Regression: noisy');

%% Part5: Linear Regression: poly_fit
load('poly_train', 'X', 'y');
load('poly_test', 'X_test', 'y_test');
w = linear_regression(X, y);
% E_train = sum(sign(w' * [ones(1, size(X, 2)); X]) ~= y) / size(X, 2);
% E_test = sum(sign(w' * [ones(1, size(X_test, 2)); X_test]) ~= y_test) / size(X_test, 2);
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);

% poly_fit with transform
X_t = [X; X(1, :).*X(2, :); X(1, :).^2; X(2, :).^2]; 
X_test_t = [X_test; X_test(1, :).*X_test(2, :); X_test(1, :).^2; X_test(2, :).^2];
w = linear_regression(X_t, y);
% E_train = sum(sign(w' * [ones(1, size(X_t, 2)); X_t]) ~= y) / size(X, 2);
% E_test = sum(sign(w' * [ones(1, size(X_test_t, 2)); X_test_t]) ~= y_test) / size(X_test, 2);
E_train = sum(sign(w' * X_t) ~= y) / size(X, 2);
E_test = sum(sign(w' * X_test_t) ~= y_test) / size(X_test, 2);

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);


%% Part6: Logistic Regression
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest = 10 * nTrain;
E_train = 0;
E_test = 0;

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain + nTest);
    y(y == -1) = 0;
    w_g = logistic(X(:, 1:nTrain), y(1:nTrain));
    % Compute training, testing error
    E_train = E_train + sum((double(sigmoid(w_g' * [ones(1, nTrain); X(:, 1:nTrain)]) > 0.5)) ~= y(1:nTrain)) / nTrain;
    E_test = E_test + sum((double(sigmoid(w_g' * [ones(1, nTest); X(:, nTrain+1:end)]) > 0.5)) ~= y(nTrain+1:end)) / nTest;
end

E_train = E_train / nRep;
E_test  = E_test / nRep;
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
y(y == 0) = -1;
plotdata(X, y, w_f, w_g, 'Logistic Regression');

%% Part7: Logistic Regression: noisy
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest = 10000; % number of training data
E_train = 0;
E_test = 0;

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain + nTest, 'noisy');
    y(y == -1) = 0;
    w_g = logistic(X(:, 1:nTrain), y(1:nTrain));
    % Compute training, testing error
    E_train = E_train + sum((double(sigmoid(w_g' * [ones(1, nTrain); X(:, 1:nTrain)]) > 0.5)) ~= y(1:nTrain)) / nTrain;
    E_test = E_test + sum((double(sigmoid(w_g' * [ones(1, nTest); X(:, nTrain+1:end)]) > 0.5)) ~= y(nTrain+1:end)) / nTest;
end

y(y == 0) = -1;
E_train = E_train / nRep;
E_test  = E_test / nRep;
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Logistic Regression: noisy');

%% Part8: SVM
nRep = 10; % number of replicates
nTrain = 100; % number of training data
nTest = 10 * nTrain;
nSV = 0;
E_train = 0;
E_test = 0;

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain + nTest);
    [w_g, num_sc] = svm(X(:, 1:nTrain), y(1:nTrain));
    % Compute training, testing error
    % Sum up number of support vectors
    E_train = E_train + sum(sign(w_g' * [ones(1, nTrain); X(:, 1:nTrain)]) ~= y(1:nTrain)) / nTrain;
    E_test = E_test + sum(sign(w_g' * [ones(1, nTest); X(:, nTrain+1:end)]) ~= y(nTrain+1:end)) / nTest;    
    nSV = nSV + num_sc;
end

E_train = E_train / nRep;
E_test  = E_test / nRep;
nSV = nSV / nRep;
fprintf('E_train is %f, E_test is %f, nSV is %f.\n', E_train, E_test, nSV);
plotdata(X(:, 1:nTrain), y(1:nTrain), w_f, w_g, 'SVM');
