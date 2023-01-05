library(tidyverse)
##### generate data ####
p <- 10
q <- 2
n <- 500
resp <- 'binary'
constrMat <- build_constrMat(p = p, q = q, struct = 'simple') # 'simple' or 'triangular'
true_load <- gen_loadings(p = p, q = q, fixed = NULL, constraint_mat = constrMat)
true_latent <- matrix(0,q,q); diag(true_latent) <- 1
#true_latent <- gen_scores(n = n, q = q, rho = 0)
true_int <- gen_intercepts(p = p, fixed = NULL, response = resp)
# manifest <- draw_data(loadings = true_load,
#                       scores = true_latent,
#                       intercepts = true_int,
#                       response = resp,
#                       seed = 1)
# manifest
manifest <- generate_data(
        SAMPLE_SIZE = n,
        LOADINGS = true_load,
        INTERCEPTS = true_int,
        LATENT_COV = true_latent,
        RESPONSE = 'binary',
        CATEGORIES = NULL,
        FAM = binomial(link = 'logit'),
        SEED = 1)
true_theta <- get_theta(TAU = true_int, LOADINGS = true_load, LATENT_COV = true_latent, CAT = rep(2,p), A = constrMat)
# ####
mod <- fit_gllvmSML(
    manifest = as.matrix(manifest),
    link = 'logit',
    constraintMat = constrMat,
    corrFLAG = 0,
    rqmcFLAG = 1,
    maxiter = 5,
    newton_tol = 1e-5,
    M = 250,
    R = 1,
    silent = F,
    ncores = 4,
    traceFLAG = 1,
    mauto_minM = 10,
    mauto_each = 1,
    mauto_tol = 1e-6)
mod
mod$z

mod$loadings
true_load
mean(
    (get_theta(mod$intercepts, mod$loadings, mod$psi, CAT = rep(2,p), A = constrMat)-
         true_theta)^2)

mean((mod$theta-true_theta)^2)

cpp_ctrl <- list(
    M = 100,
    R = 1,
    MAXITER = 1,
    NEWTON_TOL = 1e-5,
    LINK = 'logit')

fit <-  fit_gllvmSML2(
    DATA_LIST = list('DATA' = manifest,
                     'CONSTRMAT' = constrMat,
                     'CORRFLAG' = 0),
    CPP_CONTROL = cpp_ctrl,
    INIT = NULL, #true_theta,
    VERBOSEFLAG = T,
    NCORES = 4,
    RQMCFLAG = 1,
    INFOFLAG = T
)

tol <- 1e-4
sum(fit$var_rqmc<=tol)

mean((fit$theta_init-true_theta)^2)
mean((fit$theta-true_theta)^2)
tibble(init = fit$theta_init, est = fit$theta, true = true_theta) %>% print(n=100)
fit$var_rqmc
fit$time

fit$z == mod$z
fit$theta_init==mod$theta_init
fit$theta==mod$theta
fit$time; mod$time

as_tibble(fit$z) %>%
    mutate(dim = rep(paste0('dim',1:q),cpp_ctrl$R),
           seq = as.factor(sort(rep(1:cpp_ctrl$R, q)))
) %>% gather(
    key = 'point', value = 'val', starts_with('V')
) %>%
    spread(key='dim', value = 'val') %>%
    ggplot(aes(x = dim1, y = dim2, col = seq))+
    geom_point()

library(mirt)
library(tidyverse)
fitmirt <- fit_mirtEM(manifest, constrMat, corrFLAG = 0,
                      met = 'SEM', QUADPTS = 140, NCYCL = NULL)
mirt_theta <- get_theta(fitmirt$intercepts, fitmirt$loadings_std, fitmirt$psi, CAT = rep(2,p), A = constrMat)
mean((mirt_theta-true_theta)^2)
tibble(init = fit$theta_init, mirt = mirt_theta, est = fit$theta, true = true_theta) %>% print(n=100)
;
############
ISnll <- function(par){
    Lam0.vec <- par[1:p]
    Lam.vec <- par[(p+1):(p+sum(data_list$CONSTRMAT))]
    transformed_rhos_vec <- par[(p+sum(data_list$CONSTRMAT)+1):length(par)]
    if(!data_list$CORRFLAG) transformed_rhos_vec <- rep(0,q*(q-1)/2)
    #rhos_vec = (exp(2*transformed_rhos_vec)-1)/(exp(2*transformed_rhos_vec)+1)
    obj <- laplaceIS_sml2(y = data_list$DATA,
                          A = data_list$CONSTRMAT,
                          z = z,
                          lambda0 = Lam0.vec,
                          lambda = Lam.vec,
                          transformed_rhos = transformed_rhos_vec,
                          maxiter = cpp_ctrl$MAXITER,
                          tol = cpp_ctrl$NEWTON_TOL,
                          linkFLAG = 0,
                          corrFLAG = data_list$CORRFLAG,
                          grFLAG = 0,
                          ncores = 1,
                          mauto_minM = M,
                          mauto_each = 1,
                          mauto_tol = 1e-6
    )
    out <- obj$nll
    return(out)
}

# function for gradient
ISgr <- function(par){
    Lam0.vec <- par[1:p]
    Lam.vec <- par[(p+1):(p+sum(data_list$CONSTRMAT))]
    transformed_rhos_vec <- par[(p+sum(data_list$CONSTRMAT)+1):length(par)]
    obj <- laplaceIS_sml2(y = data_list$DATA,
                          A = data_list$CONSTRMAT,
                          z = z,
                          lambda0 = Lam0.vec,
                          lambda = Lam.vec,
                          transformed_rhos = transformed_rhos_vec,
                          maxiter = cpp_ctrl$MAXITER,
                          tol = cpp_ctrl$NEWTON_TOL,
                          linkFLAG = 0,
                          corrFLAG = data_list$CORRFLAG,
                          grFLAG = 1,
                          ncores = 1,
                          mauto_minM = cpp_ctrl$M,
                          mauto_each = 1,
                          mauto_tol = 1e-6
    )
    out <- obj$gradient
    return(out)
}
ISobj <- function(par){
    Lam0.vec <- par[1:p]
    Lam.vec <- par[(p+1):(p+sum(data_list$CONSTRMAT))]
    transformed_rhos_vec <- par[(p+sum(data_list$CONSTRMAT)+1):length(par)]
    #rhos_vec = (exp(2*transformed_rhos_vec)-1)/(exp(2*transformed_rhos_vec)+1)
    obj <- laplaceIS_sml2(y = data_list$DATA,
                          A = data_list$CONSTRMAT,
                          z = z,
                          lambda0 = Lam0.vec,
                          lambda = Lam.vec,
                          transformed_rhos = transformed_rhos_vec,
                          maxiter = cpp_ctrl$MAXITER,
                          tol = cpp_ctrl$NEWTON_TOL,
                          linkFLAG = 0,
                          corrFLAG = data_list$CORRFLAG,
                          grFLAG = 1,
                          ncores = 1,
                          mauto_minM = cpp_ctrl$M,
                          mauto_each = 1,
                          mauto_tol = 1e-6
    )
    out <- obj
    return(out)
}
cpp_ctrl <- list(
    M = 4,
    R = 2,
    MAXITER = 5,
    NEWTON_TOL = 1e-5,
    LINK = 'logit')


data_list <- list('DATA' = manifest, 'CONSTRMAT' = constrMat, 'CORRFLAG' = F)

rqmcFLAG <- T;
if(rqmcFLAG == T){
    cat('Drawing rqmc samples for IS.\n')
    h <- matrix(0, q*cpp_ctrl$R, cpp_ctrl$M)
    for (r in 1:cpp_ctrl$R) {

            h[(q*(r-1)+1):(q*(r)), ] <- t(rhalton(cpp_ctrl$M, q, singleseed = r))

    }
    z <- qnorm(h)
} else {
    cat('Drawing samples for IS.\n')
    z <- matrix(rnorm(q*cpp_ctrl$M), q, cpp_ctrl$M)
}
z
as_tibble(fit$z) %>%
    mutate(dim = rep(paste0('dim',1:q),cpp_ctrl$R),
           seq = as.factor(sort(rep(1:cpp_ctrl$R, q)))
    ) %>% gather(
        key = 'point', value = 'val', starts_with('V')
    ) %>%
    spread(key='dim', value = 'val') %>%
    ggplot(aes(x = dim1, y = dim2, col = seq))+
    geom_point()
ISnll(true_theta)
ISgr(true_theta)
numDeriv::grad(ISnll, true_theta)
test <- ISobj(true_theta)
tol <- 1e-4
sum(test$var_rqmc<=tol)==n

library(qrng)
t(ghalton(cpp_ctrl$M, d = q, method = 'generalized'))
