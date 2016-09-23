#ifndef UTILITY_FUNCTIONS_GPU_H
#define UTILITY_FUNCTIONS_GPU_H

#include "cuda_config.cuh"
#include <cuda_runtime.h>
#include "cublas_v2.h"

extern "C" void pi_x_product_gpu(double *x, unsigned int n, unsigned int p, double *projection);

#endif
