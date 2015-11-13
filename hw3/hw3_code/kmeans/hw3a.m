% for 3.(a)
load kmeans_data;
lSD = 0;
sSD = inf;
for i = 1:1000
    [idx, ctrs, iter_ctrs] = kmeans(X, 2);
    SD = sum(sum(bsxfun(@minus, X(idx == 1, :), ctrs(1, :)).^2, 2)) + ...
         sum(sum(bsxfun(@minus, X(idx == 2, :), ctrs(2, :)).^2, 2));
    if SD > lSD
        lSD = SD;
        lCase = struct('idx', idx, 'ctrs', ctrs, 'iter_ctrs', iter_ctrs);
    elseif SD < sSD
        sSD = SD;
        sCase = struct('idx', idx, 'ctrs', ctrs, 'iter_ctrs', iter_ctrs);
    end
end
kmeans_plot(X, lCase.idx, lCase.ctrs, lCase.iter_ctrs);
figure;
kmeans_plot(X, sCase.idx, sCase.ctrs, sCase.iter_ctrs);
