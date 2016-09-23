#' @title Measure an impact of the covariates on the response using Pearson correlation.
#' This function evaluates the Pearson correlation coefficient between the response \code{y} and each column in the design matrix \code{x} over subsamples in \code{subsamples}.
#' @param x Matrix with \code{n} observations of \code{p} covariates in each row.
#' @param y Response vector with \code{n} observations.
#' @param subsamples Matrix with \code{m} indices of \code{N} subsamples in each column. 
#' @param ... Not in use.
#' @return Numeric \code{p} by \code{N} matrix with Pearson correlations evaluated for each subsample.
#' @useDynLib rbvsGPU pearson_cor_gpu_r
#' @export

pearson.cor.gpu <- function(x,y,subsamples){
  
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  
  y <- as.numeric(y)
  storage.mode(y) <- "double"
  
  subsamples <- as.matrix(subsamples)
  storage.mode(subsamples) <- "integer"
  
  
  #check if the sizes of x and y match
  
  n <- length(y)
  if(nrow(x) != n) stop("The number of rows in 'x' must be equal to the length of 'y'.")
  
  #check for NA's
  if(any(is.na(x))) stop("Matix 'x' cannot contain NA's.")
  if(any(is.na(y))) stop("Vector 'y' cannot contain NA's.")
  if(any(is.na(subsamples))) stop("Matrix 'subsamples' cannot contain NA's.")
  
  
  #check if subsamples are actually subsamples
  min.ind <- min(subsamples)
  max.ind <- max(subsamples)
  
  if(min.ind < 1 || max.ind > n) stop("Elements of 'subsamples' must be between 1 and length(y).")
  
  return(.Call("pearson_cor_gpu_r", 
               subsamples,
               x,
               y))   

} 
