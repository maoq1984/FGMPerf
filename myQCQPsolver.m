function [alpha, weight] = myQCQPsolver(Q, b, C)

% solve p kernel matrix and q+1 variables [theta, alpha]

p = size(Q,1); % the number of base kernels
q = length(b); % not include theta

prob = [];

% the first dimension for variable theta

prob.c = [1.0; -b]; % linear term in the objective function

% Quadratical matrices, require lower triangle sparse matrix representation
prob.qcsubk = [];
prob.qcsubi = [];
prob.qcsubj = [];
prob.qcval = [];

for m=1:p
    Qml = tril(Q{m});
    % avoid non-psd
    Qml(1:q,1:q) = Qml(1:q,1:q) + eye(q) .* 1e-10;
    
    [rowm,colm,valm] = find(Qml);
    rowm = rowm + 1; % first row and column for thetha
    colm = colm + 1;
    nzm = length(valm);
    
    prob.qcsubk = [prob.qcsubk; ones(nzm,1) .* m];
    prob.qcsubi = [prob.qcsubi; rowm];
    prob.qcsubj = [prob.qcsubj; colm];
    prob.qcval = [prob.qcval; valm];    
end

% constrain linear matrix A
row = [1:p,(p+1).*ones(1,q)];
col = [ones(1,p),2:(q+1)];
val = [-ones(1,p),ones(1,q)];
prob.a = sparse(row,col,val);

% bound of the constraints
prob.blc = [];
prob.buc = [zeros(p,1);C];

% bound of the variables
prob.blx = zeros(q+1,1);
prob.bux = [];

% solver  
cmd = 'minimize echo(0)';
% cmd = 'minimize';
[rcode, res] = mosekopt(cmd,prob);

if(rcode ~= 0)
    res
end

% obtain the solution
alpha = res.sol.itr.xx(2:end);
weight = res.sol.itr.suc(1:p);

% % test results
% theta = res.sol.itr.xx(1);
% for m=1:p
%     temp = 0.5 * (alpha' * Q{m} * alpha);
%     fprintf('ok: theta = %f, temp=%f\n',theta,temp);    
% end
