function [mvy, q, obj] = find_most_violated_y_hamming(y,wxd)
% the loss is the hamming loss function

% this function is to solve the problem
%   argmax_{y' \in {-1,+1}^n } Delta(y, y') + sum_{i=1}^n y'_i wxd(i) - sum_{i=1}^n y_i wxd(i)
% RETURN:
% mvy = y', q = Delta(y, y'), objective

loss = 1 - 2 .* y .* wxd;
mvy = y;
idx = find(loss > 0); % reverse sign
mvy(idx) = - mvy(idx);

q = length(idx);

obj = sum(loss(idx));