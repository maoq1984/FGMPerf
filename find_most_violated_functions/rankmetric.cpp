#include "mex.h"
#include "common_header.h"

LABEL find_most_violated_constraint_rankmetric(LABEL y, double *score,double *q, double *valmax)
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

	if(nump)
		qsort(scorep,nump,sizeof(STRUCT_ID_SCORE),comparedown);
	if(numn)
		qsort(scoren,numn,sizeof(STRUCT_ID_SCORE),comparedown);

	STRUCT_ID_SCORE *predset=(STRUCT_ID_SCORE *)my_malloc(sizeof(STRUCT_ID_SCORE)*(ybar._totdoc+1));
  /* find max of loss(ybar,y)+score(ybar) */
	for(int i=0;i<nump;i++) {
		predset[i]=scorep[i];
		predset[i]._score-=(0.5); 
	}
	for(int i=0;i<numn;i++) {
		predset[nump+i]=scoren[i];
		predset[nump+i]._score+=(0.5); 
	}
	qsort(predset,nump+numn,sizeof(STRUCT_ID_SCORE),comparedown);

	long sump=0;
	long sumn=0;
	for(int i=0;i<numn+nump;i++) {
		if(y._class[predset[i]._id] > 0) {
			ybar._class[predset[i]._id]=((numn-2.0*sumn)*0.5*100.0/nump)/numn;
			sump++;
		}
		else {
			ybar._class[predset[i]._id]=-((nump-2.0*(nump-sump))*0.5*100.0/nump)/numn;
			sumn++;
		}
	}

	*q = swappedpairs_loss(y,ybar);
	*valmax = 0;
	for(int i = 0;i<ybar._totdoc;i++) *valmax += (ybar._class[i] * score[i]);

	free(scorep);
	free(scoren);
	free(predset);

	return ybar;
}


// input two vectors
// prhs[0] = y, column vector
// prhs[1] = wxd

// output
// plhs[0] = mvy
// plhs[1] = q
// plhs[2] = obj

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if(nrhs != 2)
	{
		mexErrMsgTxt("the parameters: [mvy,q,obj] = rankmetric(y, wxd)\n");
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

	LABEL ybar = find_most_violated_constraint_rankmetric(ylabel,wxd,q,obj);

	for(int i = 0;i<ybar._totdoc;i++) mvy[i] = ybar._class[i];
	delete []ybar._class;
	delete []ylabel._class;
}