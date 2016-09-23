#include "utility_functions.h"

int unsigned_int_cmp(const void *aa, const void *bb)
{

	const unsigned int *a = aa, *b = bb;
	return (*a < *b) ? -1 : (*a > *b);

}

int is_in_array(unsigned int *sorted, unsigned int element, unsigned int length)
{

	unsigned int *item =
	    (unsigned int *)bsearch(&element, sorted, length,
				    sizeof(unsigned int), unsigned_int_cmp);
	if (item != NULL)
		return 1;
	else
		return 0;

}

SEXP matrix_projection_in_place_gpu_r(SEXP x, SEXP projection, SEXP active){
  SEXP x_new, x_dim;
  
  PROTECT(x_dim = getAttrib(x, R_DimSymbol));
  
  register unsigned int n = INTEGER(x_dim)[0];
  register unsigned int p = INTEGER(x_dim)[1];
  register unsigned int n_active = length(active);
  
  PROTECT(x_new = allocMatrix(REALSXP, n,p-n_active));
  
  matrix_projection_in_place_gpu(REAL(x_new), REAL(x), n, p, REAL(projection), (unsigned int *)INTEGER(active), n_active);
  
  UNPROTECT(2);
  
  return x_new;
}


void pi_x_product_gpu(double *x, unsigned int n, unsigned int p, double *projection);


void matrix_projection_in_place_gpu(double *x_new, double *x, unsigned int n, unsigned int p, double *projection, unsigned int *active, unsigned int n_active){
  
  register unsigned int i, j, k=0;
  double alpha = 1.0;
  double beta = 0.0;
  size_t bytes_in_column = sizeof(double) *n;
  

  
  unsigned int *active_cpy = Calloc(n_active, unsigned int);
  memcpy(active_cpy, active, n_active * sizeof(unsigned int));
  qsort(active_cpy, n_active, sizeof(unsigned int), unsigned_int_cmp);
  
  //select relevant columns
  for(j=0; j<p; j++) if(is_in_array(active_cpy, j+1, n_active)==0){
    memcpy(&x_new[k*n],&x[j*n],bytes_in_column);
    k++;
  } 
  
  Free(active_cpy);

  pi_x_product_gpu(x_new, n, k, projection);

  
}

