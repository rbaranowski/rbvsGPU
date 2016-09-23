#ifndef PEARSON_COR_GPU_H
#define PEARSON_COR_GPU_H

#include "cuda_config.cuh"

__global__ void pearson_cor_kernel(double *x, double *y, unsigned int *subsamples, double *cor, unsigned int n, unsigned int p, unsigned int m, unsigned int B);
extern "C" void pearson_cor_vector_gpu(unsigned int *subsamples, unsigned int m, unsigned int B, double *x, unsigned int n, unsigned int p, double *y, double *cor);

#endif
