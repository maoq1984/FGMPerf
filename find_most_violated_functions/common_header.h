
#ifndef _COMMON_HEADER_H_
#define _COMMON_HEADER_H_

#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>

#define MAX(x,y)      ((x) < (y) ? (y) : (x))
#define MIN(x,y)      ((x) > (y) ? (y) : (x))
#define SIGN(x)       ((x) > (0) ? (1) : (((x) < (0) ? (-1) : (0))))

/* Identifiers for loss functions */
#define ZEROONE      0
#define FONE         1
#define ERRORRATE    2
#define PRBEP        3
#define PREC_K       4
#define REC_K        5
#define SWAPPEDPAIRS 10
#define AVGPREC      11

struct LABEL {
  double *_class; /* vector of labels */
  int _totdoc;    /* size of set */
};


struct STRUCT_ID_SCORE {
	int _id;
	double _score;
	double _tiebreak;
} ;

void   *my_malloc(size_t); 
int compareup(const void *a, const void *b);
int comparedown(const void *a, const void *b);

double zeroone(int a, int b, int c, int d);
double fone(int a, int b, int c, int d);
double errorrate(int a, int b, int c, int d);
double prec(int a, int b, int c, int d);
double rec(int a, int b, int c, int d);
double swappedpairs(LABEL y, LABEL ybar);
double rocarea(LABEL y, LABEL ybar);
double prbep(LABEL y, LABEL ybar);
double avgprec(LABEL y, LABEL ybar);

double zeroone_loss(int a, int b, int c, int d);
double fone_loss(int a, int b, int c, int d);
double errorrate_loss(int a, int b, int c, int d);
double prbep_loss(int a, int b, int c, int d);
double prec_k_loss(int a, int b, int c, int d);
double rec_k_loss(int a, int b, int c, int d);
double swappedpairs_loss(LABEL y, LABEL ybar);
double avgprec_loss(LABEL y, LABEL ybar);


#endif