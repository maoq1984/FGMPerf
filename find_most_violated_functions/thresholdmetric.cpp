#include "mex.h"
#include "common_header.h"

LABEL find_most_violated_constraint_thresholdmetric(LABEL y, double *score,int loss_function,
													double prec_rec_k_frac, double *q, double *valmax)
{

	LABEL ybar;
	ybar._totdoc=y._totdoc;
	ybar._class=(double *)my_malloc(sizeof(double)*y._totdoc);

	int nump = 0;
	int numn = 0;
	STRUCT_ID_SCORE *scorep=(STRUCT_ID_SCORE *)my_malloc(sizeof(STRUCT_ID_SCORE)*(ybar._totdoc+1));
	STRUCT_ID_SCORE *scoren=(STRUCT_ID_SCORE *)my_malloc(sizeof(STRUCT_ID_SCORE)*(ybar._totdoc+1));

	for(int i=0;i<y._totdoc;i++)
	{
		if(y._class[i] > 0)
		{
			scorep[nump]._score = score[i];
			scorep[nump]._tiebreak=0;
			scorep[nump]._id=i;
			nump++;
		}
		else {
			scoren[numn]._score=score[i];
			scoren[numn]._tiebreak=0;
			scoren[numn]._id=i;
			numn++;
		}		
	}

	double *sump=(double *)my_malloc(sizeof(double)*(ybar._totdoc+1));
	double *sumn=(double *)my_malloc(sizeof(double)*(ybar._totdoc+1));

	if(nump)
		qsort(scorep,nump,sizeof(STRUCT_ID_SCORE),comparedown);
	sump[0]=0;
	for(int i=0;i<nump;i++) {
		sump[i+1]=sump[i]+scorep[i]._score;
	}
	if(numn)
		qsort(scoren,numn,sizeof(STRUCT_ID_SCORE),compareup);
	sumn[0]=0;
	for(int i=0;i<numn;i++) {
		sumn[i+1]=sumn[i]+scoren[i]._score;
	}

	int threshp=0,threshn=0;

	double loss;
	int start=1;
	int prec_rec_k=(int)(nump*prec_rec_k_frac);
	if(prec_rec_k<1) prec_rec_k=1;
	for(int a=0;a<=nump;a++) {
		for(int d=0;d<=numn;d++) {
			if(loss_function == ZEROONE)
				loss=zeroone_loss(a,numn-d,nump-a,d);
			else if(loss_function == FONE)
				loss=fone_loss(a,numn-d,nump-a,d);
			else if(loss_function == ERRORRATE)
				loss=errorrate_loss(a,numn-d,nump-a,d);
			else if((loss_function == PRBEP) && (a+numn-d == nump))
				loss=prbep_loss(a,numn-d,nump-a,d);
			else if((loss_function == PREC_K) && (a+numn-d >= prec_rec_k))
				loss=prec_k_loss(a,numn-d,nump-a,d);
			else if((loss_function == REC_K) && (a+numn-d <= prec_rec_k)) 
				loss=rec_k_loss(a,numn-d,nump-a,d);
			else {
				loss=0;
			}
			if(loss > 0) {
				double val=loss+sump[a]-(sump[nump]-sump[a])-sumn[d]+(sumn[numn]-sumn[d]);

				if((val > *valmax) || (start)) {
					start=0;
					*valmax=val;
					threshp=a;
					threshn=d;
					*q = loss;
				}
			}
		}
	}

	for(int i=0;i<nump;i++) {
		if(i<threshp) 
			ybar._class[scorep[i]._id]=y._class[scorep[i]._id];
		else 
			ybar._class[scorep[i]._id]=-y._class[scorep[i]._id];
	}
	for(int i=0;i<numn;i++) {
		if(i<threshn) 
			ybar._class[scoren[i]._id]=y._class[scoren[i]._id];
		else 
			ybar._class[scoren[i]._id]=-y._class[scoren[i]._id];
	}

	free(scorep);
	free(scoren);
	free(sump);
	free(sumn);

	return ybar;
}


// input two vectors
// prhs[0] = y, column vector
// prhs[1] = wxd
// prhs[2] = loss_function
// prhs[3] = prec_rec_k_frac

// output
// plhs[0] = mvy
// plhs[1] = q
// plhs[2] = obj

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if(nrhs <= 2)
	{
		mexErrMsgTxt("the parameters: [mvy,q,obj] = thresholdmetric(y, wxd, loss_function, prec_rec_k_frac)\n");
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

	int loss_function = 0;
	double prec_rec_k_frac = 0.0;
	if(nrhs > 2)
	{
		loss_function = (int) mxGetScalar(prhs[2]);
		if(nrhs > 3)	
			prec_rec_k_frac = (int) mxGetScalar(prhs[3]);
	}
	LABEL ylabel;
	ylabel._class = new double[m];
	for(int i = 0;i<m;i++)
	{
		ylabel._class[i] = y[i];
	}
	ylabel._totdoc = m;


	plhs[0] = mxCreateDoubleMatrix(m,1,mxREAL);
	double *mvy = mxGetPr(plhs[0]);

	plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
	double *q = mxGetPr(plhs[1]);

	plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
	double *obj = mxGetPr(plhs[2]);

	LABEL ybar = find_most_violated_constraint_thresholdmetric(ylabel,wxd,loss_function,prec_rec_k_frac,q,obj);

	for(int i = 0;i<ybar._totdoc;i++) mvy[i] = ybar._class[i];
	delete []ybar._class;
	delete []ylabel._class;
}