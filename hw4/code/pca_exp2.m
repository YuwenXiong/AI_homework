load('ORL_data', 'fea_Train', 'gnd_Train', 'fea_Test', 'gnd_Test');

% YOUR CODE HERE
k_for_knn = 5;
[vec, ~] = LDA(gnd_Train, [], fea_Train);
reduced_fea_train = fea_Train * vec;
reduced_fea_test = fea_Test * vec;
y = knn(reduced_fea_test', reduced_fea_train', gnd_Train', k_for_knn);
fprintf('error rate %f\n', sum(y' ~= gnd_Test) / length(gnd_Test));
