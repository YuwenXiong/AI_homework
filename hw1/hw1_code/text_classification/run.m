%ham_train contains the occurrences of each word in ham emails. 1-by-N vector
ham_train = csvread('ham_train.csv');
%spam_train contains the occurrences of each word in spam emails. 1-by-N vector
spam_train = csvread('spam_train.csv');
%N is the size of vocabulary.
N = size(ham_train, 2);
%There 9034 ham emails and 3372 spam emails in the training samples
num_ham_train = 9034;
num_spam_train = 3372;
%Do smoothing
x = [ham_train;spam_train] + 1;

%ham_test contains the occurences of each word in each ham test email. P-by-N vector, with P is number of ham test emails.
load ham_test.txt;
ham_test_tight = spconvert(ham_test);
ham_test = sparse(size(ham_test_tight, 1), size(ham_train, 2));
ham_test(:, 1:size(ham_test_tight, 2)) = ham_test_tight;
%spam_test contains the occurences of each word in each spam test email. Q-by-N vector, with Q is number of spam test emails.
load spam_test.txt;
spam_test_tight = spconvert(spam_test);
spam_test = sparse(size(spam_test_tight, 1), size(spam_train, 2));
spam_test(:, 1:size(spam_test_tight, 2)) = spam_test_tight;

%TODO
%Implement a ham/spam email classifier, and calculate the accuracy of your classifier
%training
[C, N] = size(x);
total = num_ham_train + num_spam_train;
l = likelihood(x);
prior = [num_ham_train / total, num_spam_train / total]';

%testing
% px = zeros(C, N);
% for i = 1:N
% 	px(:, i) = l(:, i) .* prior;
% end
% px = sum(px, 1);

% p = zeros(C, , N);
% for i = 1:N
%     p(:, i) = l(:, i).^x(i) .* prior / px(i);
% end


test_num = [size(ham_test, 1), size(spam_test, 1)]; 
TP = 0;
FP = 0;
TN = 0;
FN = 0;

for i = 1:test_num(1)
    a = find(ham_test(i, :) ~= 0);
    temp = sum([ham_test(i, a) .* log(l(1, a)); ham_test(i, a) .* log(l(2, a))], 2) + [log(prior(1)); log(prior(2))];
%     temp = sum(log([l(1, a).^(nonzeros(ham_test(i, :))'); l(2, a).^(nonzeros(ham_test(i, :))')]), 2) + [log(prior(1)); log(prior(2))];
%     temp = sum(log(l(1, a).^(nonzeros(ham_test(i, :))') ./ l(2, a).^(nonzeros(ham_test(i, :))'))) + log(prior(1)) - log(prior(2));
    
    if temp(1) < temp(2) %temp < 0
        FP = FP + 1;
    else
        TN = TN + 1;
    end
end


for i = 1:test_num(2)
    a = find(spam_test(i, :) ~= 0);
    temp = sum([spam_test(i, a) .* log(l(1, a)); spam_test(i, a) .* log(l(2, a))], 2) + [log(prior(1)); log(prior(2))];
%     temp = sum(log(l(1, a).^(nonzeros(spam_test(i, :))') ./ l(2, a).^(nonzeros(spam_test(i, :))'))) + log(prior(1)) - log(prior(2));
    if temp(1) < temp(2) %temp < 0
        TP = TP + 1;
    else
        FN = FN + 1;
    end
end

[v, idx] = sort(l(1, :) ./ l(2, :));
top = idx(1:10);

