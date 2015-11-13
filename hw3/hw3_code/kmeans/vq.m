img = imread('sample0.jpg');
fea = double(reshape(img, size(img, 1)*size(img, 2), 3));
imshow(img);
figure;
% YOUR (TWO LINE) CODE HERE
[idx, ctrs, ~] = kmeans(fea, 64);
fea = ctrs(idx, :);
imshow(uint8(reshape(fea, size(img))));
