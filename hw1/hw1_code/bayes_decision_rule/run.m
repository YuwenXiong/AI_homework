% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%%load data
load('data');
all_x = cat(2, x1_train, x1_test, x2_train, x2_test);
range = [min(all_x), max(all_x)];
train_x = get_x_distribution(x1_train, x2_train, range);
test_x = get_x_distribution(x1_test, x2_test, range);

%% Part1 likelihood: 
l = likelihood(train_x);

bar(range(1):range(2), l');
xlabel('x');
ylabel('P(x|\omega)');
axis([range(1) - 1, range(2) + 1, 0, 0.5]);

%TODO
%compute the number of all the misclassified x using maximum likelihood decision rule
res = 0;
for i = 1:size(test_x, 2)
    if (l(1, i) > l(2, i)) 
        res = res + test_x(2, i);
    else
        res = res + test_x(1, i);
    end
end

%% Part2 posterior:
p = posterior(train_x);

bar(range(1):range(2), p');
xlabel('x');
ylabel('P(\omega|x)');
axis([range(1) - 1, range(2) + 1, 0, 1.2]);

%TODO
%compute the number of all the misclassified x using optimal bayes decision rule
res2 = 0;
for i = 1:size(test_x, 2)
    if p(1, i) > p(2, i)
        res2 = res2 + test_x(2, i);
    elseif p(1, i) < p(2, i)
        res2 = res2 + test_x(1, i);
%     else
%         if l(1, i) > l(2, i)
%             res2 = res2 + test_x(2, i);
%         elseif l(1, i) < l(2, i)
%             res2 = res2 + test_x(1, i);
%         else
%             if train_x(1, i) > train_x(2, i)
%                 res2 = res2 + test_x(2, i);
%             else
%                 res2 = res2 + test_x(1, i);
%             end
%         end
    end
end


%% Part3 risk:
risk = [0, 1; 2, 0];
%TODO
%get the minimal risk using optimal bayes decision rule and risk weights
R = 0;
for i = 1:size(test_x, 2)
    r1 = risk(1, 1) * p(1, i) + risk(1, 2) * p(2, i);
    r2 = risk(2, 1) * p(1, i) + risk(2, 2) * p(2, i);
    if r1 < r2
        R = R + r1;% * test_x(2, i);
    else
        R = R + r2;% * test_x(1, i);
    end
end
