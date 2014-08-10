#include "mex.h"
#include "common_header.h"

double  loss(LABEL y, LABEL ybar, int loss_function)
{
  /* loss for correct label y and predicted label ybar. The loss for
     y==ybar has to be zero. loss_function is set with the -l option. */
  int a=0,b=0,c=0,d=0,i;
  double loss=1;

  /* compute contingency table */
  for(i=0;i<y._totdoc;i++) {
    if((y._class[i] > 0) && (ybar._class[i] > 0)) {
      a++;
    }
    if((y._class[i] > 0) && (ybar._class[i] <= 0)) {
      c++;
    }
    if((y._class[i] < 0) && (ybar._class[i] > 0)) {
      b++;
    }
    if((y._class[i] < 0) && (ybar._class[i] <= 0)) {
      d++;
    }
    /* printf("%f %f\n",y.class[i],ybar.class[i]); */
  }
  /* Return the loss according to the selected loss function. */
  if(loss_function == ZEROONE) { /* type 0 loss: 0/1 loss */
                                  /* return 0, if y==ybar. return 1 else */
    loss=zeroone_loss(a,b,c,d);
  }
  else if(loss_function == FONE) {
    loss=fone_loss(a,b,c,d);
  }
  else if(loss_function == ERRORRATE) {
    loss=errorrate_loss(a,b,c,d);
  }
  else if(loss_function == PRBEP) {
    /* WARNING: only valid if called for a labeling that is at PRBEP */
    loss=prbep_loss(a,b,c,d);
  }
  else if(loss_function == PREC_K) {
    /* WARNING: only valid if for a labeling that predicts k positives */
    loss=prec_k_loss(a,b,c,d);
  }
  else if(loss_function == REC_K) {
    /* WARNING: only valid if for a labeling that predicts k positives */
    loss=rec_k_loss(a,b,c,d);
  }
  else if(loss_function == SWAPPEDPAIRS) {
    loss=swappedpairs_loss(y,ybar);
  }
  else if(loss_function == AVGPREC) {
    loss=avgprec_loss(y,ybar);
  }
  else {
    /* Put your code for different loss functions here. But then
       find_most_violated_constraint_???(x, y, sm) has to return the
       highest scoring label with the largest loss. */
    printf("Unknown loss function type: %d\n",loss_function);
    exit(1);
  }
  return(loss);
}

double eval_prediction(LABEL y, LABEL ypred,int loss_function)
{
	if(loss_function == ERRORRATE)
		return loss(y,ypred,loss_function);
	else if(loss_function == PREC_K || loss_function == REC_K || loss_function == FONE)
		return 100.0 - loss(y,ypred,loss_function);
	else if(loss_function == PRBEP)
		return prbep(y,ypred);
	else if(loss_function == SWAPPEDPAIRS)
		return rocarea(y,ypred);
	else
		return avgprec(y,ypred);
}


// input two vectors
// prhs[0] = y, column vector
// prhs[1] = wxd
// prhs[2] = loss_function

// output
// plhs[0] = multivariate measurement value

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if(nrhs != 3)
	{
		mexErrMsgTxt("the parameters: [mvy,q,obj] = rankmetric(y, wxd,loss_function)\n");
	}

	double *y = mxGetPr(prhs[0]);
	int n = mxGetN(prhs[0]);
	int m = mxGetM(prhs[0]);

	double *wxd = mxGetPr(prhs[1]);
	int n1 = mxGetN(prhs[1]);
	int m1 = mxGetM(prhs[1]);

	if(n != 1 || n1 != 1)
	{
		mexErrMsgTxt("both vectors are N x 1 dimensional column vector\n");
	}

	int loss_function = (int) mxGetScalar(prhs[2]);

	LABEL ylabel, ypred;
	ylabel._class = new double[m];
	ypred._class = new double[m];
	for(int i = 0;i<m;i++)
	{
		ylabel._class[i] = y[i];
		ypred._class[i] = wxd[i];
	}
	ylabel._totdoc = m;
	ypred._totdoc = m;


	plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
	double *mm = mxGetPr(plhs[0]);

	*mm = eval_prediction(ylabel, ypred,loss_function);

	delete []ypred._class;
	delete []ylabel._class;
}