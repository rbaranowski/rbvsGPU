#' @title  Generate factor model design matrix.
#' @description This function enables a quick generation of random design matrices simulating data on GPU.
#' @param n Number of independent realisations of the factor model.
#' @param p Number of covariates.
#' @param n.factors Number of factors.
#' @param sigma Standard deviation for the normal distribution.
#' @return \code{n} by \code{p} matrix with independent rows following factor model (see details).
#' @details The elements of the matrix returned by this routine satisfy \eqn{X_{ij} = \sum_{l=1}^{n.factors} f_{ijl} \varphi_{il} + \theta_{ij}}{X_{ij} = \sum_{l=1}^{K} f_{ijl} \varphi_{il} + \theta_{ij}}
#' with \eqn{f_{ijl}}{f_{ijl}}, \eqn{\varphi_{il}}{\varphi_{il}}, \eqn{\theta_{ij}}{\theta_{ij}}, \eqn{\varepsilon_{i}}{\varepsilon_{i}}  i.i.d. \eqn{\mathcal{N}(0,(sigma)^2)}{\mathcal{N}(0,(sigma)^2)}.
#' @useDynLib rbvsGPU factor_model_gpu_r
#' @export
#' @details The elements of the matrix returned by this routine satisfy \eqn{X_{ij} = \sum_{l=1}^{n.factors} f_{ijl} \varphi_{il} + \theta_{ij}}{X_{ij} = \sum_{l=1}^{K} f_{ijl} \varphi_{il} + \theta_{ij}}
#' with \eqn{f_{ijl}}{f_{ijl}}, \eqn{\varphi_{il}}{\varphi_{il}}, \eqn{\theta_{ij}}{\theta_{ij}}, \eqn{\varepsilon_{i}}{\varepsilon_{i}}  i.i.d. \eqn{\mathcal{N}(0,(sigma)^2)}{\mathcal{N}(0,(sigma)^2)}.


gen.factor.model.design.gpu <- function(n,p, n.factors, sigma=1){
  
  n <- as.integer(n)
  p <- as.integer(p)
  n.factors <- as.integer(n.factors)
  sigma <- as.double(sigma)
  
  if(any(is.na(n))) stop("'n' cannot be NA.")
  if(any(is.na(p))) stop("'p' cannot be NA.")
  if(any(is.na(n.factors))) stop("'n.factors' cannot be NA.")
  if(any(is.na(sigma))) stop("'n.factors' cannot be NA.")
  
  if(length(n) < 1) stop("'n' cannot be empty.")
  if(length(p) < 1) stop("'p' cannot be empty.")
  if(length(n.factors) < 1) stop("'n.factors' cannot be empty.")
  if(length(sigma) < 1) stop("'sigma' cannot be empty.")
  
  if(length(n) > 1){
    warning("'n' should be a scalar. Only first element will be used.")
    n <- n[1]
  } 
  
  if(length(p) > 1) {
    warning("'p' should be a scalar. Only first element will be used.")
    p <- p[1]
  }
  
  if(length(n.factors) > 1){
    warning("'n.factors' should be a scalar. Only first element will be used.")
    n.factors <- n.factors[1]
  } 
  if(length(sigma) > 1) {
    warning("'sigma' should be a scalar. Only first element will be used.")
    sigma <- sigma[1]
  }
  
  if(n <= 0)  stop("'n' must be > 0.")
  if(p <= 0)  stop("'p' must be > 0.")
  if(n.factors < 0)  stop("'n.factors' must be >= 0.")
  if(sigma <= 0)  stop("'sigma' must be > 0.")
  
  return(.Call("factor_model_gpu_r", as.integer(n), as.integer(p), as.integer(n.factors)))
}
