function img = hack_pca(filename)
% Input: filename -- input image file name/path
% Output: img -- image without rotation

img_r = double(imread(filename));

% YOUR CODE HERE
[x, y] = find(img_r < 128);
X = [x, y];

[vec, ~] = pca(X);
% imshow(uint8(img_r * vec(:, 1)));
% img = ones(size(img_r));
% img(sub2ind(size(img), X(:, 1), X(:, 2))) = 0;
img = imrotate(uint8(img_r), -atan(vec(1, 1) / vec(1, 2)) / pi * 180);
end