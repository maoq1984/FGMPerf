mex -O -largeArrayDims -c common_header.cpp
mex -O -largeArrayDims  thresholdmetric.cpp common_header.obj
mex -O -largeArrayDims rankmetric.cpp common_header.obj
mex -O -largeArrayDims eval_prediction.cpp common_header.obj