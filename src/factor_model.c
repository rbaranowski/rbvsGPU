#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>

void factor_model_gpu(double *x, unsigned int n, unsigned int p, unsigned int k, unsigned long long int seed);


SEXP factor_model_gpu_r(SEXP n, SEXP p, SEXP n_factors){
  
  unsigned int val_n = INTEGER(n)[0];
  unsigned int val_p = INTEGER(p)[0];
  unsigned int val_n_factors = INTEGER(n_factors)[0];
  unsigned long long int seed;
  

  SEXP x = PROTECT(allocMatrix(REALSXP,val_n,val_p));
  
  double *ptr_x = REAL(x);
  
  GetRNGstate();

  seed = (unsigned long long int) (INT_MAX * unif_rand());
  
  factor_model_gpu(ptr_x, val_n, val_p, val_n_factors, seed);

  PutRNGstate();

  
  UNPROTECT(1);
  
  return(x);
}
