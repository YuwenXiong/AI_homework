load('TDT2_data', 'fea', 'gnd');

% YOUR CODE HERE
s_accu = zeros(100, 1);
k_accu = zeros(100, 1);
s_Mihat = zeros(100, 1);
k_Mihat = zeros(100, 1);
for num = 1:100
    W = full(constructW(fea));
    idx = spectral(W, length(unique(gnd)));
    [a, ~] = hist(idx, unique(idx));
    [~, I] = sort(a, 'descend');
    ugnd = unique(gnd);
    for i = 5:-1:1
        idx(idx == I(i)) = ugnd(i);
    end

    kidx = litekmeans(fea, length(unique(gnd)));
    [a, b] = hist(kidx, unique(kidx));
    [~, I] = sort(a, 'descend');
    for i = 5:-1:1
        kidx(kidx == I(i)) = ugnd(i);
    end

    s_accu(num) = sum(idx == gnd) / length(gnd);
    k_accu(num) = sum(kidx == gnd) / length(gnd);
    s_Mihat(num) = MutualInfo(gnd, idx);
    k_Mihat(num) = MutualInfo(gnd, kidx);
end
spectral_accu = mean(s_accu);
kmeans_accu = mean(k_accu);
spectral_Mihat = mean(s_Mihat);
kmeans_Mihat = mean(k_Mihat);
