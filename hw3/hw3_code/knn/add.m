function [X, y] = add(X, y, tX, v)
X = [X double(tX)]; 
y = [y; v'];