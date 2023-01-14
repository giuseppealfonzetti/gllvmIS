
<!-- README.md is generated from README.Rmd. Please edit that file -->

# gllvmSML

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![CRAN
status](https://www.r-pkg.org/badges/version/gllvmSML)](https://CRAN.R-project.org/package=gllvmSML)
<!-- badges: end -->

The gllvmSML deals with the estimation of GLLVMs using Similuted Maximum
Likelihood. By now, only estimation for binary data is implemented.

## Installation

You can install the development version of glllvmSML like so:

``` r
devtools::install_github("giuseppealfonzetti/gllvmIS")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(gllvmSML)

# Setup the model
p <- 10
q <- 2
n <- 250
resp <- 'binary'
constrMat <- build_constrMat(p = p, q = q, struct = 'simple') # 'simple' or 'triangular'
true_load <- gen_loadings(p = p, q = q, fixed = NULL, constraint_mat = constrMat)
true_latent <- matrix(0,q,q); diag(true_latent) <- 1
true_int <- gen_intercepts(p = p, fixed = NULL, response = resp)
true_theta <- get_theta(TAU = true_int, LOADINGS = true_load, LATENT_COV = true_latent, CAT = rep(2,p), A = constrMat)

# generate data
manifest <- generate_data(
        SAMPLE_SIZE = n,
        LOADINGS = true_load,
        INTERCEPTS = true_int,
        LATENT_COV = true_latent,
        RESPONSE = 'binary',
        CATEGORIES = NULL,
        FAM = binomial(link = 'logit'),
        SEED = 1)


# fit the model
cpp_ctrl <- list(
    M = 100,
    R = 1,
    MAXITER = 5,
    NEWTON_TOL = 1e-5,
    LINK = 'logit')

fit <-  fit_gllvmSML2(
    DATA_LIST = list('DATA' = manifest,
                     'CONSTRMAT' = constrMat,
                     'CORRFLAG' = 0),
    CPP_CONTROL = cpp_ctrl,
    INIT = NULL, 
    VERBOSEFLAG = T,
    RQMCFLAG = 2,
    INFOFLAG = T
)
#> 1. Initialising at default values
#> 2. Drawing rqmc Halton (qrng::ghalton).
#> 3. Optimisation...
#> 4. Done! (80.38 secs)

mean((fit$theta_init-true_theta)^2)
#> [1] 0.1523643
mean((fit$theta-true_theta)^2)
#> [1] 0.06165599
```
