#include "common_header.h"

void *my_malloc(size_t size)
{
	void *ptr;
	if(size<=0) size=1; /* for AIX compatibility */
	ptr=(void *)malloc(size);
	if(!ptr) { 
		perror ("Out of memory!\n"); 
		exit (1); 
	}
	return(ptr);
}

int compareup(const void *a, const void *b) 
{
	double va,vb;
	va=((STRUCT_ID_SCORE *)a)->_score;
	vb=((STRUCT_ID_SCORE *)b)->_score;
	if(va == vb) {
		va=((STRUCT_ID_SCORE *)a)->_tiebreak;
		vb=((STRUCT_ID_SCORE *)b)->_tiebreak;
	}
	return((va > vb) - (va < vb));
}

int comparedown(const void *a, const void *b) 
{
	return(-compareup(a,b));
}

/*------- Performance measures --------*/

double zeroone(int a, int b, int c, int d) 
{
  if((a+d) == (a+b+c+d)) 
    return(0.0);
  else
    return(1.0);
}

double fone(int a, int b, int c, int d) 
{
  if((a == 0) || (a+b == 0) || (a+c == 0)) return(0.0);
  double precision=prec(a,b,c,d);
  double recall=rec(a,b,c,d);
  return(2.0*precision*recall/(precision+recall));
}

double prec(int a, int b, int c, int d) 
{
  /* Returns precision as fractional value. */
  if((a+b) == 0) return(0.0);
  return((double)a/(double)(a+b));
}

double rec(int a, int b, int c, int d) 
{
  /* Returns recall as fractional value. */
  if((a+c) == 0) return(0.0);
  return((double)a/(double)(a+c));
}

double errorrate(int a, int b, int c, int d) 
{
  /* Returns number of errors. */
  if((a+b+c+d) == 0) return(0.0);
  return(((double)(b+c))/(double)(a+b+c+d));
}

double swappedpairs(LABEL y, LABEL ybar)
{
  /* Returns percentage of swapped pos/neg pairs (i.e. 100 - ROC Area) for
     prediction vectors that encode the number of misranked examples
     for each particular example. */
  /* WARNING: Works only for labels in the compressed representation */
  int i;
  double sum=0;
  for(i=0;i<y._totdoc;i++) 
    sum+=fabs(y._class[i]-ybar._class[i]);
  return(sum/2.0);
}

double rocarea(LABEL y, LABEL ybar)
{
  /* Returns ROC Area for ybar containing scores that define a ranking
     of examples. Breaks ties in ranking pessimistically. */
  long i,nump,numn;
  double swappedpairs;
  STRUCT_ID_SCORE *predset;

  predset=(STRUCT_ID_SCORE *)my_malloc(sizeof(STRUCT_ID_SCORE)*(ybar._totdoc+1));
  for(i=0;i<ybar._totdoc;i++) {
    predset[i]._score=ybar._class[i];
    predset[i]._tiebreak=-y._class[i];
    predset[i]._id=i;
  }
  qsort(predset,ybar._totdoc,sizeof(STRUCT_ID_SCORE),comparedown);
  numn=0;
  nump=0;
  swappedpairs=0;
  for(i=0;i<ybar._totdoc;i++) {
    if(y._class[predset[i]._id] > 0) {
      swappedpairs+=numn;
      nump++;
    }
    else {
      numn++;
    }
  }
  free(predset);
  return(100.0-100.0*swappedpairs/((double)numn)/((double)nump));
}

double prbep(LABEL y, LABEL ybar)
{
  /* Returns PRBEP for ybar containing scores that define a ranking
     of examples. Breaks ties in ranking pessimistically. */
  long i,nump,a;
  STRUCT_ID_SCORE *predset;

  predset=(STRUCT_ID_SCORE *)my_malloc(sizeof(STRUCT_ID_SCORE)*(ybar._totdoc+1));
  nump=0;
  for(i=0;i<ybar._totdoc;i++) {
    predset[i]._score=ybar._class[i];
    predset[i]._tiebreak=-y._class[i];
    predset[i]._id=i;
    if(y._class[i] > 0) 
      nump++;
 }
  qsort(predset,ybar._totdoc,sizeof(STRUCT_ID_SCORE),comparedown);
  a=0;
  for(i=0;i<nump;i++) {
    if(y._class[predset[i]._id] > 0) {
      a++;
    }
  }
  free(predset);
  return(100.0*prec(a,nump-a,0,0));
}

double avgprec_compressed(LABEL y, LABEL ybar)
{
  /* Returns Average Precision for y and ybar in compressed
     representation (also see avgprec()). Breaks ties in ranking
     pessimistically. */
  int i,ii,nump,numn,a,b;
  double apr;
  STRUCT_ID_SCORE *predset;

  nump=0;
  numn=0;
  for(i=0;i<ybar._totdoc;i++) {
    if(y._class[i] > 0) 
      nump++;
    else 
      numn++;
  }
  /*  printf("nump=%d, numn=%d\n", nump, numn); */

  ii=0;
  predset=(STRUCT_ID_SCORE *)my_malloc(sizeof(STRUCT_ID_SCORE)*(nump+1));
  for(i=0;i<ybar._totdoc;i++) {
    if(y._class[i] > 0) {
      predset[ii]._score=ybar._class[i];
      predset[ii]._tiebreak=-y._class[i];
      predset[ii]._id=i;
      ii++;
    }
  }
  qsort(predset,nump,sizeof(STRUCT_ID_SCORE),comparedown);

  apr=0;
  for(a=1;a<=nump;a++) {
    b=(int)(numn-predset[a-1]._score)/2;
    /* printf("negabove[%d]=%d,",a,b); */
    apr+=prec(a,b,0,0);
  }

  free(predset);

  return(100.0*(apr/(double)(nump)));
}

double avgprec(LABEL y, LABEL ybar)
{
  /* Returns Average Precision for ybar containing scores that define a ranking
     of examples. Breaks ties in ranking pessimistically. */
  long i,nump,numn;
  double apr;
  STRUCT_ID_SCORE *predset;

  predset=(STRUCT_ID_SCORE *)my_malloc(sizeof(STRUCT_ID_SCORE)*(ybar._totdoc+1));
  for(i=0;i<ybar._totdoc;i++) {
    predset[i]._score=ybar._class[i];
    predset[i]._tiebreak=-y._class[i];
    predset[i]._id=i;
  }
  qsort(predset,ybar._totdoc,sizeof(STRUCT_ID_SCORE),comparedown);
  numn=0;
  nump=0;
  apr=0;
  for(i=0;i<ybar._totdoc;i++) {
    if(y._class[predset[i]._id] > 0) {
      nump++;
      apr+=prec(nump,numn,0,0);
    }
    else {
      numn++;
    }
  }
  free(predset);
  return(100.0*(apr/(double)(nump)));
}

/*------- Loss functions based on performance measures --------*/

double zeroone_loss(int a, int b, int c, int d) 
{
  return(zeroone(a,b,c,d));
}

double fone_loss(int a, int b, int c, int d) 
{
  return(100.0*(1.0-fone(a,b,c,d)));
}

double errorrate_loss(int a, int b, int c, int d) 
{
  return(100.0*errorrate(a,b,c,d));
}

double prbep_loss(int a, int b, int c, int d) 
{
  /* WARNING: Returns lower bound on PRBEP, if b!=c. */
  double precision=prec(a,b,c,d);
  double recall=rec(a,b,c,d);
  if(precision < recall) 
    return(100.0*(1.0-precision));
  else
    return(100.0*(1.0-recall));
}

double prec_k_loss(int a, int b, int c, int d) 
{
  /* WARNING: Only valid if called with a+c==k. */
  return(100.0*(1.0-prec(a,b,c,d)));
}

double rec_k_loss(int a, int b, int c, int d) 
{
  /* WARNING: Only valid if called with a+c==k. */
  return(100.0*(1.0-rec(a,b,c,d)));
}

double swappedpairs_loss(LABEL y, LABEL ybar)
{  
  double nump=0,numn=0;
  long i;
  for(i=0;i<y._totdoc;i++) {
    if(y._class[i] > 0) 
      nump++;
    else 
      numn++;
  }
  /*  return(100.0*swappedpairs(y,ybar)/(nump*numn)); */
  return(swappedpairs(y,ybar));
}

double avgprec_loss(LABEL y, LABEL ybar)
{
  return(100.0-avgprec_compressed(y,ybar));
}