#include "factor_model_gpu.cuh"

__global__ void add_one_factor_kernel(double *x, unsigned int n, unsigned int p, double *loadings, double *noise){

	
	register unsigned int row_id = blockIdx.x*MAX_BLOCKS + blockIdx.y;

	if(row_id < n){
		
		__shared__ double loading;

		if(threadIdx.x==0){
			loading = loadings[row_id];
		} 
		__syncthreads();

		register unsigned int col_id = threadIdx.x;
		unsigned long long int shift = MAX_THREADS * n;
		unsigned long long int l = col_id * n + row_id;
		double xij, eps;

		while(col_id < p){

			xij = x[l];
			eps = noise[l];

			xij += loading * eps; 
			x[l] = xij;

			l += shift; 
			col_id += MAX_THREADS;
		
		}
	
	}

}

__global__ void add_noise_kernel(double *x, unsigned int n, unsigned int p, double *noise){

	register unsigned int row_id = blockIdx.x*MAX_BLOCKS + blockIdx.y;

	if(row_id < n){
	

		register unsigned int col_id = threadIdx.x;
		unsigned long long int shift = MAX_THREADS * n;
		unsigned long long int l = col_id * n + row_id;
		double xij, eps;

		while(col_id < p){

			xij = x[l];
			eps = noise[l];

			xij += eps; 
			x[l] = xij;

			l += shift; 
			col_id += MAX_THREADS;
		
		}
	
	}

}


extern "C" void factor_model_gpu(double *x, unsigned int n, unsigned int p, unsigned int k, unsigned long long int seed){

	double *d_x, *d_loadings, *d_noise;
	register unsigned int i; 

	cudaError_t err;
	curandGenerator_t generator;
	
 
	//allocate memory on GPU TODO: check for errors
	cudaMalloc( (void**)&d_x, n * p * sizeof(double));
	cudaMemset(d_x, 0,  n * p * sizeof(double));

	cudaMalloc( (void**)&d_loadings, n * sizeof(double));
	cudaMalloc( (void**)&d_noise, n * p * sizeof(double));


	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator, seed);
	

	dim3 grid_blocks(n/MAX_BLOCKS+1, MAX_BLOCKS);
	dim3 grid_threads(MAX_THREADS);

	for(i=0; i<k; i++){

		//generate loadings
		curandGenerateNormalDouble(generator, d_loadings, n, 0.0, 1.0);
		//generate noise
		curandGenerateNormalDouble(generator, d_noise, n*p, 0.0, 1.0);

		//add factor
		add_one_factor_kernel<<<grid_blocks, grid_threads >>>(d_x, n, p, d_loadings, d_noise);

	}

	//generate noise
	curandGenerateNormalDouble(generator, d_noise, n*p, 0.0, 1.0);
	//add noise
	add_noise_kernel<<<grid_blocks, grid_threads >>>(d_x, n, p, d_noise);

	//transfer x to the host
	cudaMemcpy(x, d_x, n * p * sizeof(double), cudaMemcpyDeviceToHost);

	//free memory
	cudaFree(d_x);
	cudaFree(d_noise);
	cudaFree(d_loadings);
	
	curandDestroyGenerator(generator);

	err = cudaGetLastError();
	if (err != cudaSuccess) printf("Cuda error: %s\n", cudaGetErrorString(err));
}
