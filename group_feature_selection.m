function [beta, alpha, weight] = group_feature_selection(X,y,C,D,T,options)
% training instance (x_i,y_i), X: nxd, y: nx1
% tradeoff parameter C
% stop criterion eps
% the set of groups, cell D={d_1,....d_T}, each d_1 is an index vector for features
% maximal iterations, mmaxiter
eps = options.eps2;
maxiter = options.maxiter2;


time = cputime;
% T = size(D,1); % the number of groups
[n,dim] = size(X);

mvY = zeros(n,maxiter); % store all the most violated y vector

Q = cell(T,1); % allocate memory for QCQP
for t=1:T
    Q{t} = zeros(maxiter,maxiter); 
end

qset = []; % keep the value 1/n Delta(y,y')

% initializing with w = 0
wxd = zeros(n,1); % keep the value sum_{t=1}^T <w_t,x_i .* d^t>
[mvy, q] = find_most_violated_y(y,wxd,options);

% the main loop
iter = 1;
K = 1; % the number of cuts
min_fval = 0;
while 1
    mvY(:,K) = mvy; % add one of the most violated found currently
    
    % calculate the Q matrix, Q^t_{K,k}, for k=1:K, and b_K
    for t=1:T
        ptK = - X(:,D{t})' * (y - mvy);
        for k=1:K
            ptk = - X(:,D{t})' * (y - mvY(:,k));
            Q{t}(K,k) = ptK' * ptk ./ (n^2);
            Q{t}(k,K) = Q{t}(K,k);
        end
    end
    qset = [qset;q/n];
    
    %solve QCQP problem
    [alpha, weight] = myQCQPsolver(Q,qset,C);
    
     % prune cuts
    scale_alpha = alpha ./ C;
    prune_idx = find(scale_alpha < 1e-6);
    if(~isempty(prune_idx) && length(prune_idx) < K)
        remain_idx = find(scale_alpha >= 1e-6);
        red_K = length(remain_idx);

        temp_mvY = mvY(1:n,remain_idx);
        mvY(1:n,1:K) = zeros(n,K);
        mvY(1:n,1:red_K) = temp_mvY;

        for t=1:T
            temp_Qt = Q{t}(remain_idx,remain_idx);
            Q{t}(1:K,1:K) = zeros(K,K);
            Q{t}(1:red_K,1:red_K) = temp_Qt;
        end

        qset = qset(remain_idx);

        alpha = alpha(remain_idx);
        
        K = red_K;
    end
    
    % compute regulariation term
    reg = 0;
    for t=1:T
        Qt = Q{t}(1:K,1:K);
        reg = reg + weight(t) * sqrt(alpha' * Qt * alpha);
    end
    reg = 0.5 * reg^2;
    
    % compute pieve-wise linear approximate objective, the lowe bound
    linobj = zeros(K,1);
    for k=1:K
        for t=1:T
           Qt = Q{t}(1:K,1:K);
           linobj(k) = linobj(k) - weight(t) * (alpha' * Qt(:,k));
        end
        linobj(k) = linobj(k) + qset(k);
    end
    linobj = reg + C * linobj;
    lower_bound = max([linobj; 0]);
    
    % calculate the intermedian values and compute wxd, and find most violated mvy
    beta = zeros(n,1);
    for j=1:n
        beta(j) = sum(alpha) .*y(j) - mvY(j,1:K) * alpha;
    end
        
    wxd = zeros(n,1);
    for t=1:T
        wxd = wxd + weight(t) .* X(:,D{t}) *( X(:,D{t})' * beta );
    end
    wxd = wxd ./ n;

    t1 = cputime;
    [mvy,q,loss] = find_most_violated_y(y,wxd,options);
    t1 = cputime - t1;
    upper_bound = reg + C * loss / n;
    if K==1
        min_fval = upper_bound;
    else
        if min_fval > upper_bound
            min_fval = upper_bound;
        end
    end
    
%     gap = min_fval - lower_bound;
    gap = (min_fval - lower_bound)/min_fval;
%     fprintf('T = %d iter: %d, Ncut=%d gap=%f,t1=%f\n',T, iter, K, gap,t1);
    % stop criterion
    if( gap <= eps || iter >= maxiter)
        time = cputime - time;
        fprintf('gap = %f, upper_bound=%f,cost = %f, Ncut=%d ',gap,min_fval,time,K);
        if(iter < maxiter)
            fprintf('total %d iterates to reach eps-optimal\n',iter);
        else
            fprintf('reach maximal iterates\n');
        end
        break;
    end
    
    K = K + 1;
    iter = iter + 1;
end

