#include "pearson_cor.h"

SEXP pearson_cor_gpu_r(SEXP subsamples, SEXP x, SEXP y){
  
  SEXP x_dim, subsamples_dim, cor;
  
  PROTECT(x_dim = getAttrib(x, R_DimSymbol));
  PROTECT(subsamples_dim = getAttrib(subsamples, R_DimSymbol));
  
  register unsigned int n = INTEGER(x_dim)[0];
  register unsigned int p = INTEGER(x_dim)[1];
  register unsigned int m = INTEGER(subsamples_dim)[0];
  register unsigned int B = INTEGER(subsamples_dim)[1];
  register unsigned int i,j, k;
  
  
  PROTECT(cor = allocMatrix(REALSXP,p,B));
  
  double *ptr_x = REAL(x);
  double *ptr_y = REAL(y);
  double *ptr_cor = REAL(cor); 
  unsigned int *ptr_subsamples = (unsigned int *) INTEGER(subsamples);
  

  //determine batch size 

  register unsigned int n_batches = ((n>B ? n : B) * p * sizeof(double) -1)/ MAX_CUDA_DATA_SIZE +1;
  register unsigned int p_batch = p / n_batches;
  register unsigned int p_calculated = 0;

  double *tmp_cor = Calloc(p_batch * B, double);
  
  for(k=0; k<n_batches; k++){

      pearson_cor_vector_gpu(ptr_subsamples, m, B, &ptr_x[p_calculated * n], n, p_batch, ptr_y, tmp_cor);

      #pragma omp parallel for
      for(j=0; j<p_batch; j++){
        for(i=0; i<B; i++) ptr_cor[i*p+p_calculated+j] = tmp_cor[i*p_batch + j];
      }

      p_calculated += p_batch;
      
  }
 
  if(p_calculated < p){
      
      p_batch = p-p_calculated;
      pearson_cor_vector_gpu(ptr_subsamples, m, B, &ptr_x[p_calculated * n], n, p_batch, ptr_y, tmp_cor);

      #pragma omp parallel for
      for(j=0; j<p_batch; j++){
        for(i=0; i<B; i++) ptr_cor[i*p+p_calculated+j] = tmp_cor[i*p_batch + j];
      }

  }

  

  UNPROTECT(3);
  Free(tmp_cor);

  return cor;
  
}

