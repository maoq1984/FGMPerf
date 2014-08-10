function acc = eval_Acc(ytest, fval)
acc = 100 * sum(sign(fval) == ytest) / length(ytest);