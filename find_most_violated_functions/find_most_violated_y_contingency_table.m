function [mvy, q, obj] = find_most_violated_y_contingency_table(y,wxd,loss_type,prec_rec_k_frac)
% k = prec_rec_k_frac * num of positive instances
% defining the identities in file common_header.h
% #define ZEROONE      0
% #define FONE         1
% #define ERRORRATE    2
% #define PRBEP        3
% #define PREC_K       4
% #define REC_K        5
% #define SWAPPEDPAIRS 10
% #define AVGPREC      11


if strcmp(loss_type,'prec_k')
    loss_function = 4;
elseif strcmp(loss_type,'rec_k')
    loss_function = 5;
elseif strcmp(loss_type,'fone')
    loss_function = 1;
elseif strcmp(loss_type, 'prbep')
    loss_function = 3;
end

[mvy,q,temp_obj] = thresholdmetric(y,wxd,loss_function,prec_rec_k_frac);
obj = temp_obj - y' * wxd;

