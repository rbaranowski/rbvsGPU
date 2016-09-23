#include "pearson_cor_gpu.cuh"

__global__ void pearson_cor_kernel(double *x, double *y, unsigned int *subsamples, double *cor, unsigned int n, unsigned int p, unsigned int m, unsigned int B){
	  
	register unsigned int col_id = blockIdx.y*MAX_BLOCKS + blockIdx.x;
	 
	if(col_id < p){  
 
		register unsigned int rows_id = blockIdx.z*MAX_THREADS + threadIdx.x;
		
		if(rows_id < B){
		
			unsigned int i; 
			unsigned int k;
			extern __shared__ double s_y[];
			double *s_xj = &s_y[n];

			unsigned int *rows = &subsamples[rows_id * m];
			double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x_sq = 0.0, sum_y_sq = 0.0;
			__shared__ double *xj;
	 

			if(threadIdx.x == 0) xj = &x[col_id * n];
			__syncthreads();

			i = threadIdx.x;

			while(i < n){

					s_xj[i] = xj[i];
					s_y[i] = y[i];
					i += blockDim.x;
			}
			
			__syncthreads();

			double yk, xjk; 
			
			for(i=0; i<m; i++){

				k = rows[i]-1;

				yk =  s_y[k];
				xjk = s_xj[k];

				sum_y += yk;
				sum_y_sq += yk*yk;
				sum_x += xjk;
				sum_xy += xjk * yk;
				sum_x_sq += xjk * xjk;


			}
	 

		   double sd =  sqrt((m*sum_y_sq - sum_y * sum_y)*(m*sum_y_sq - sum_y * sum_y));
	       

		   if(sd > DBL_EPSILON) cor[rows_id * p + col_id]  = fabs((m*sum_xy - (sum_x * sum_y))/(sd));
		   else  cor[rows_id * p + col_id] = 0.0;

	   }
	}
}

extern "C" void pearson_cor_vector_gpu(unsigned int *subsamples, unsigned int m, unsigned int B, double *x, unsigned int n, unsigned int p, double *y, double *cor){

	unsigned int *d_subsamples;
	double *d_x, *d_y, *d_cor;
	cudaError_t err;
 
	//allocate memory on GPU TODO: check for errors
	cudaMalloc( (void**)&d_subsamples, m * B * sizeof(unsigned int));
	cudaMalloc( (void**)&d_x, n * p * sizeof(double));
	cudaMalloc( (void**)&d_y, n * sizeof(double));
	cudaMalloc( (void**)&d_cor, B * p * sizeof(double));
	//transfer data to GPU
	cudaMemcpy(d_subsamples, subsamples,m * B * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, n * p * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice);

	//compute correlations
	/*
	dim3 grid(p/THREADS_PER_BLOCK +1, B);  
	pearson_cor_kernel<<<grid,THREADS_PER_BLOCK>>>(d_x, d_y, d_subsamples, d_cor, n, p, m, B);
	*/ 

	dim3 grid_blocks(MAX_BLOCKS, p/MAX_BLOCKS+1, B/MAX_THREADS+1);
	dim3 grid_threads(MAX_THREADS);

	pearson_cor_kernel<<<grid_blocks, grid_threads, sizeof(double) * n * 2 >>>(d_x, d_y, d_subsamples, d_cor, n, p, m, B);

	err = cudaGetLastError(); 
	if (err != cudaSuccess) printf("Cuda error: %s\n", cudaGetErrorString(err));
	//transfer correlations back
	cudaMemcpy(cor, d_cor, B * p * sizeof(double), cudaMemcpyDeviceToHost);

	//Free memory
	cudaFree(d_subsamples);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_cor);

	err = cudaGetLastError();
	if (err != cudaSuccess) printf("Cuda error: %s\n", cudaGetErrorString(err));
}
