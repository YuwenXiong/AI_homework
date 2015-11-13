function X_norm = normalize(X)
X_norm = zeros(size(X));
for i = 1:size(X, 1)
    X_norm(:, i) = (X(:, i) - mean(X(:, i))) / std(X(:, i));
end