load('ORL_data', 'fea_Train', 'gnd_Train', 'fea_Test', 'gnd_Test');

% YOUR CODE HERE

% 1. Feature preprocessing
% 2. Run PCA
% 3. Visualize eigenface
% 4. Project data on to low dimensional space
% 5. Run KNN in low dimensional space
% 6. Recover face images form low dimensional space, visualize them
data = [fea_Train; fea_Test];
[vec, ~] = pca(data);
show_face(vec');

% [U, S, V] = mySVD(data, 64);
% show_face(U * V');

k = [8 16 32 64 128 1024];
k_for_knn = 1;
for i = k
    reduced_fea_train = fea_Train * vec(:, 1:i);
    reduced_fea_test = fea_Test * vec(:, 1:i);
%     [U, ~, V] = mySVD(data, i);
%     reduced_fea_train = fea_Train * V;
%     reduced_fea_test = fea_Test * V;
    figure;
    show_face(reduced_fea_test(1:40, :) * vec(:, 1:i)');
%     show_face(reduced_fea_test(1:40, :) * V');
    y = knn(reduced_fea_test', reduced_fea_train', gnd_Train', k_for_knn);
    fprintf('error rate for k = %d: %f\n', i, sum(y' ~= gnd_Test) / length(gnd_Test));
end
