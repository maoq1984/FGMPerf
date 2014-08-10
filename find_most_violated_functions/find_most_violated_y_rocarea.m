function [mvy, q, obj] = find_most_violated_y_rocarea(y,wxd)
[mvy, q, temp_obj] = rankmetric(y,wxd);
obj = temp_obj - y' * wxd;