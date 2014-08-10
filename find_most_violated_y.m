function [mvy, q, obj] = find_most_violated_y(y,wxd,options)
% this function is to solve the problem
%   argmax_{y' \in {-1,+1}^n } Delta(y, y') + sum_{i=1}^n y'_i wxd(i) - sum_{i=1}^n y_i wxd(i)
% RETURN:
% mvy = y', q = Delta(y, y'), objective
if ~isfield(options,'loss_type')
    options.loss_type = 'hamming';
end

if ~isfield(options,'prec_rec_k_frac')
    options.prec_rec_k_frac = 2; % two times of positive instances
end

loss_type = options.loss_type;
if strcmp(loss_type,'hamming')
    [mvy, q, obj] = find_most_violated_y_hamming(y,wxd);
elseif strcmp(loss_type,'fone') || strcmp(loss_type,'prec_k') ...
        ||strcmp(loss_type,'rec_k') ||strcmp(loss_type,'prbep')
    [mvy, q, obj] = find_most_violated_y_contingency_table(y,wxd,loss_type,options.prec_rec_k_frac);
elseif strcmp(loss_type,'swappedpairs')
    [mvy, q, obj] = find_most_violated_y_rocarea(y,wxd);
end