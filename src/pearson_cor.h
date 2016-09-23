#ifndef PEARSON_COR_H
#define PEARSON_COR_H

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#define MAX_CUDA_DATA_SIZE 756000000

SEXP pearson_cor_gpu_r(SEXP subsamples, SEXP x, SEXP y);
void pearson_cor_vector_gpu(unsigned int *subsamples, unsigned int m, unsigned int B, double *x, unsigned int n, unsigned int p, double *y, double *cor);

#endif 
