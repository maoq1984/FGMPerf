This code is the implementation of the following paper:

Qi Mao, Ivor W. Tsang. A Feature Selection Method for Multivariate Performance Measures. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(9): 2051 - 2063, Sept 2013.

Any bugs in the code, please email to Qi Mao (maoq1984@gmail.com). 

1. System Requirement

Matlab with mex setup by choosing C++ compiler
Mosek toolbox from http://www.mosek.com/

The code is mostly tested in 32bit and 64bit Windows System. For other systems, it may still be ok, but without guarantee.

2. Compiling

Open Matlab, and enter into the folder "find_most_violated_functions"
> make

3. Optimized measures

In this code, the following measures are implemented:
(1) hamming: accuracy
(2) fone : f1
(3) prec_k : precision @ k
(4) rec_k : recall @ k
(5) prbep : precision/recall breakeven point

4. Parameter Setting

options structure
  .eps1: termination condition of outer loop (group feature generation)
  .maxiter1 : maximum iteration of outer loop
  .eps2: termination condition of inner loop (group feature selection)
  .maxiter1: maximum iteration of inner loop
  .loss_type: the above five types
  
Another two parameters: 
B: the budget parameter
C: tradeoff parameter scaled by the number of training instances.
NOTE: the terminate condition and the maximum iteration in the outer loop would affect the number of selected features and time complexity, as well as performance. 

5. Test Example

> test_fgmperf('URL1',0.1)

This example implements all five measures on dataset 'URL1.mat'.
To make the running successful, the toolbox of Mosek should be added correctly into the current workspace of Matlab .

More details about the data format and parameters settings are referred to this test example.
