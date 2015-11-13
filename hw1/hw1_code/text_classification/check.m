function f = check(x, l, prior)

N = size(x, 2);
px = zeros(2, N);
for i = 1:N
	px(:, i) = l(:, i).^x(i) .* prior;
end
px = sum(px, 1);
p = zeros(2, N);
for i = 1:N
    p(:, i) = l(:, i).^x(i) .* prior / px(i);
end
f = sum(log(p(1, :))) < sum(log(p(2, :)));
