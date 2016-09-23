#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>

SEXP matrix_projection_in_place_gpu_r(SEXP x, SEXP projection, SEXP active);
void matrix_projection_in_place_gpu(double *x_new, double *x, unsigned int n, unsigned int p, double *projection, unsigned int *active, unsigned int n_active);
int is_in_array(unsigned int *sorted, unsigned int element, unsigned int length);
int unsigned_int_cmp(const void *aa, const void *bb);

#endif
