function test_fgmperf(filename,c_ratio)

% filename = 'URL1';
% c_ratio = 0.1;

addpath('find_most_violated_functions');
addpath('C:\Program Files\Mosek\6\toolbox\r2007a');

fprintf('---------process %s \n---',filename);

load(filename);
xapp=Xtrain; yapp=Ytrain;
xtest=Xtest; ytest=Ytest;

clear Xtrain Ytrain Xtest Ytest

C = c_ratio * length(yapp); % tradeoff parameter
normalize_data = 0;
normalize_kernel = 0;

% set parameters
options.eps1 = 0.1;
options.maxiter1 = 10;
options.eps2 = 0.01;
options.maxiter2 = 500;


rep = 1;
prop = [2 5 10 20 30 40 50 60];
prop_size = length(prop);

% store data
num_algo = 5;
results = cell(num_algo,1);
for i= 1:num_algo
    results{i}.time = zeros(rep,prop_size);
    results{i}.accuracy = zeros(rep,prop_size);
    results{i}.fone = zeros(rep,prop_size);
    results{i}.prec_k = zeros(rep,prop_size);
    results{i}.rec_k = zeros(rep,prop_size);
    results{i}.prbep = zeros(rep,prop_size);
    
    results{i}.groups = cell(rep,prop_size);
    results{i}.weight = cell(rep,prop_size);
    results{i}.beta = cell(rep,prop_size);
end

for i=1:rep
    if normalize_data
        [xapp,xtest] = normalizemeanstd(xapp,xtest);
    end
    [n,dim]= size(xapp);
    
    for m=1:prop_size
        B = prop(m);
        test_pos_size = sum(ytest == 1);
        pre_rec_k = 2 * test_pos_size;
        fprintf('-------process %dth iterate, B=%d C=%f\n-------',i,B,C);
        
        % hamming loss performance measurement        
        fprintf('------------hamming loss----------\n');
        options.loss_type = 'hamming';
        t1 = cputime;
        if normalize_kernel
            [beta,weight,D,WEIGHT] = group_feature_generation_normalizeK(xapp, yapp, C, B, options);
            weight = weight .* WEIGHT;
        else
            [beta,weight,D] = group_feature_generation(xapp, yapp, C, B, options);
        end
        results{1}.time(i,m) = cputime - t1;
        
        fval = zeros(length(ytest),1);
        ng = length(weight);
        for t=1:ng
            fval = fval + weight(t) .* xtest(:,D{t}) *(xapp(:,D{t})' * beta);
        end
        
        results{1}.accuracy(i,m) = eval_Acc(ytest,fval);
        results{1}.fone(i,m) = eval_performance(ytest,fval,'fone');
        
        [sort_val,sort_idx] = sort(fval,'descend');
        new_fval = - ones(length(ytest),1);
        new_fval(sort_idx(1:pre_rec_k)) = 1;        
        
        results{1}.prec_k(i,m) = eval_performance(ytest,new_fval,'prec_k');
        results{1}.rec_k(i,m) = eval_performance(ytest,new_fval,'rec_k');
        results{1}.prbep(i,m) = eval_performance(ytest,fval,'prbep');        
        results{1}.groups{i,m} = D;
        results{1}.weight{i,m} = weight;
        results{1}.beta{i,m} = beta; 
                
        fprintf('time=%f\n',results{1}.time(i,m));
        fprintf('accuracy=%f\n',results{1}.accuracy(i,m));
        fprintf('fone=%f\n',results{1}.fone(i,m));
        fprintf('prec_k=%f\n',results{1}.prec_k(i,m));
        fprintf('rec_k=%f\n',results{1}.rec_k(i,m));
        fprintf('prbep=%f\n',results{1}.prbep(i,m));
       
        %% multivariate performance
        %% f1-score performance measurement      
        fprintf('------------f1 score----------\n');
        options.loss_type = 'fone';
        
        t2 = cputime;
        [beta,weight,D] = group_feature_generation(xapp, yapp, C/100, B, options);
        results{2}.time(i,m) = cputime - t2;
        
        fval = zeros(length(ytest),1);
        ng = length(weight);
        for t=1:ng
            fval = fval + weight(t) .* xtest(:,D{t}) *(xapp(:,D{t})' * beta);
        end
        
        results{2}.accuracy(i,m) = eval_Acc(ytest,fval);
        results{2}.fone(i,m) = eval_performance(ytest,fval,'fone');
        
        [sort_val,sort_idx] = sort(fval,'descend');
        new_fval = - ones(length(ytest),1);
        new_fval(sort_idx(1:pre_rec_k)) = 1;     
        
        results{2}.prec_k(i,m) = eval_performance(ytest,new_fval,'prec_k');
        results{2}.rec_k(i,m) = eval_performance(ytest,new_fval,'rec_k');
        results{2}.prbep(i,m) = eval_performance(ytest,fval,'prbep');        
        results{2}.groups{i,m} = D;
        results{2}.weight{i,m} = weight;
        results{2}.beta{i,m} = beta; 
                
        fprintf('time=%f\n',results{2}.time(i,m));
        fprintf('accuracy=%f\n',results{2}.accuracy(i,m));
        fprintf('fone=%f\n',results{2}.fone(i,m));
        fprintf('prec_k=%f\n',results{2}.prec_k(i,m));
        fprintf('rec_k=%f\n',results{2}.rec_k(i,m));
        fprintf('prbep=%f\n',results{2}.prbep(i,m));
        
        %% prec_k performance measurement
        options.loss_type = 'prec_k';
        fprintf('---------prec_k loss----------\n');
        t1 = cputime;
        [beta,weight,D] = group_feature_generation(xapp, yapp, C/100, B, options);
        results{3}.time(i,m) = cputime - t1;
        
        fval = zeros(length(ytest),1);
        ng = length(weight);
        for t=1:ng
            fval = fval + weight(t) .* xtest(:,D{t}) *(xapp(:,D{t})' * beta);
        end
        
        results{3}.accuracy(i,m) = eval_Acc(ytest,fval);
        results{3}.fone(i,m) = eval_performance(ytest,fval,'fone');
        
        [sort_val,sort_idx] = sort(fval,'descend');
        new_fval = - ones(length(ytest),1);
        new_fval(sort_idx(1:pre_rec_k)) = 1; 
        
        results{3}.prec_k(i,m) = eval_performance(ytest,new_fval,'prec_k');
        results{3}.rec_k(i,m) = eval_performance(ytest,new_fval,'rec_k');
        results{3}.prbep(i,m) = eval_performance(ytest,fval,'prbep');        
        results{3}.groups{i,m} = D;
        results{3}.weight{i,m} = weight;
        results{3}.beta{i,m} = beta; 
                
        fprintf('time=%f\n',results{3}.time(i,m));
        fprintf('accuracy=%f\n',results{3}.accuracy(i,m));
        fprintf('fone=%f\n',results{3}.fone(i,m));
        fprintf('prec_k=%f\n',results{3}.prec_k(i,m));
        fprintf('rec_k=%f\n',results{3}.rec_k(i,m));
        fprintf('prbep=%f\n',results{3}.prbep(i,m));
        
        
        %% rec_k performance measurement
        options.loss_type = 'rec_k';
        fprintf('---------rec_k loss----------\n');
        t1 = cputime;
        [beta,weight,D] = group_feature_generation(xapp, yapp, C/100, B, options);
        results{4}.time(i,m) = cputime - t1;
        
        fval = zeros(length(ytest),1);
        ng = length(weight);
        for t=1:ng
            fval = fval + weight(t) .* xtest(:,D{t}) *(xapp(:,D{t})' * beta);
        end
        
        results{4}.accuracy(i,m) = eval_Acc(ytest,fval);
        results{4}.fone(i,m) = eval_performance(ytest,fval,'fone');
        
        [sort_val,sort_idx] = sort(fval,'descend');
        new_fval = - ones(length(ytest),1);
        new_fval(sort_idx(1:pre_rec_k)) = 1; 
        
        results{4}.prec_k(i,m) = eval_performance(ytest,new_fval,'prec_k');
        results{4}.rec_k(i,m) = eval_performance(ytest,new_fval,'rec_k');
        results{4}.prbep(i,m) = eval_performance(ytest,fval,'prbep');        
        results{4}.groups{i,m} = D;
        results{4}.weight{i,m} = weight;
        results{4}.beta{i,m} = beta; 
                
        fprintf('time=%f\n',results{4}.time(i,m));
        fprintf('accuracy=%f\n',results{4}.accuracy(i,m));
        fprintf('fone=%f\n',results{4}.fone(i,m));
        fprintf('prec_k=%f\n',results{4}.prec_k(i,m));
        fprintf('rec_k=%f\n',results{4}.rec_k(i,m));
        fprintf('prbep=%f\n',results{4}.prbep(i,m));
        
        
        %% prbep performance measurement
        options.loss_type = 'prbep';
        fprintf('---------prbep loss----------\n');
        t1 = cputime;
        [beta,weight,D] = group_feature_generation(xapp, yapp, C/100, B, options);
        results{5}.time(i,m) = cputime - t1;
        
        fval = zeros(length(ytest),1);
        ng = length(weight);
        for t=1:ng
            fval = fval + weight(t) .* xtest(:,D{t}) *(xapp(:,D{t})' * beta);
        end
        
        results{5}.accuracy(i,m) = eval_Acc(ytest,fval);
        results{5}.fone(i,m) = eval_performance(ytest,fval,'fone');
        
        [sort_val,sort_idx] = sort(fval,'descend');
        new_fval = - ones(length(ytest),1);
        new_fval(sort_idx(1:pre_rec_k)) = 1; 
        
        results{5}.prec_k(i,m) = eval_performance(ytest,new_fval,'prec_k');
        results{5}.rec_k(i,m) = eval_performance(ytest,new_fval,'rec_k');
        results{5}.prbep(i,m) = eval_performance(ytest,fval,'prbep');        
        results{5}.groups{i,m} = D;
        results{5}.weight{i,m} = weight;
        results{5}.beta{i,m} = beta; 
                
        fprintf('time=%f\n',results{5}.time(i,m));
        fprintf('accuracy=%f\n',results{5}.accuracy(i,m));
        fprintf('fone=%f\n',results{5}.fone(i,m));
        fprintf('prec_k=%f\n',results{5}.prec_k(i,m));
        fprintf('rec_k=%f\n',results{5}.rec_k(i,m));
        fprintf('prbep=%f\n',results{5}.prbep(i,m));
    end
end

save_file = sprintf('%s_C%d.mat',filename,C);
save(save_file,'results');
