function [beta,weight,D] = group_feature_generation(X, y, C, B, options)
% 
%
eps1 = options.eps1;
maxiter1 = options.maxiter1;

[n,dim] = size(X); 

D = cell(maxiter1,1);

% initalization
beta = zeros(n,1);
d = find_most_violated_d(X,beta,B);


T = 1;
iter = 1;
while 1
    D{T} = d;
    
    % do group model selection
    [beta, alpha, weight] = group_feature_selection(X,y,C,D,T,options);
    
    % prune groups according to weight
    prune_idx = find(weight < 1e-9);
    if ~isempty(prune_idx)
        remain_idx = find(weight >= 1e-9);
        red_T = length(remain_idx);
        
        for t=1:red_T
            D{t} = D{remain_idx(t)};
        end
        for t=(red_T+1) : T
            D{t} = [];
        end
        
        weight = weight(remain_idx);
        T = red_T;
    end
    
    % compute the lower bound
    linobj = zeros(T,1);
    for t=1:T
        temp = beta' * X(:,D{t});
       linobj(t) = (temp * temp') / n^2;
    end
    lower_bound = max(linobj);
    
    [d, obj] = find_most_violated_d(X,beta,B);
    upper_bound = obj / n^2;
    f_min = upper_bound;
    
    gap = (f_min - lower_bound) / f_min;
    fprintf('iter: %d, T = %d gap = %f, f_min = %f, lower_bound = %f\n',iter,T,gap,f_min,lower_bound);
    
    if gap < eps1 || iter > maxiter1
        if(iter < maxiter1)
            fprintf('reach eps-optimal\n');
        else
            fprintf('reach the maximal iterates\n');
        end
        break;
    end
    
    T = T + 1;
    iter = iter + 1;
end


function [d, obj] = find_most_violated_d(X,beta,B)
c = X' * beta;
weights = c.^2;
[sort_val,sort_idx] = sort(weights,'descend');
d = sort_idx(1:B);
obj = sum(weights(d));
