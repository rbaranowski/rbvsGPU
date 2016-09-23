#include "utility_functions_gpu.cuh"


void pi_x_product_gpu(double *x, unsigned int n, unsigned int p, double *projection){
	
	double *d_x, *d_projection, *d_result;
	cudaError_t err;

	cudaMalloc( (void**)&d_x, n * p * sizeof(double));
	cudaMalloc( (void**)&d_result, n * p * sizeof(double));
	cudaMalloc( (void**)&d_projection, n * n * sizeof(double));

	cudaMemcpy(d_x, x, n * p * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_projection, projection, n * n * sizeof(double), cudaMemcpyHostToDevice);
   	
   	cublasHandle_t handle;
	cublasCreate(&handle);




	double alpha = 1.0;
	double beta = 0.0;

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, p, n, &alpha, d_projection, n, d_x, n, &beta, d_result, n);
	
	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("Cuda error: %s\n", cudaGetErrorString(err));
	
	cudaMemcpy(x, d_result, n * p * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_x);
	cudaFree(d_result);
	cudaFree(d_projection);
	cublasDestroy(handle);


	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("Cuda error: %s\n", cudaGetErrorString(err));


}
