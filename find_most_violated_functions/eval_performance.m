function value = eval_performance(y,wxd,loss_type)

if strcmp(loss_type,'prec_k')
    loss_function = 4;
elseif strcmp(loss_type,'rec_k')
    loss_function = 5;
elseif strcmp(loss_type,'fone')
    loss_function = 1;
elseif strcmp(loss_type, 'prbep')
    loss_function = 3;
elseif strcmp(loss_type, 'swappedpairs')
    loss_function = 10;
elseif strcmp(loss_type,'avgprec')
    loss_function = 11;
end

value = eval_prediction(y,wxd,loss_function);