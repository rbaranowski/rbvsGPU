#ifndef FACTOR_MODEL_GPU_H
#define FACTOR_MODEL_GPU_H

#include "cuda_config.cuh"

__global__ void	add_one_factor_kernel(double *x, unsigned int n, unsigned int p, double *loadings, double *noise);
__global__ void	add_noise_kernel(double *x, unsigned int n, unsigned int p, double *noise);
extern "C" void factor_model_gpu(double *x, unsigned int n, unsigned int p, unsigned int k, unsigned long long seed);

#endif
