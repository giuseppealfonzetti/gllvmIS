source('examples/generate_test_data.R')

p <- ncol(manifest)
q <- 2
n <- nrow(manifest)
constrMat <- build_constrMat(p = p, q = q, struct = 'simple') # 'simple' or 'triangular'

lambda0_init = qlogis(colMeans(manifest))
lambda_init = rep(0.5, sum(constrMat))
transformed_rhos_init = rep(0, q*(q-1)/2)

R <- 1; M <- 100
h <- matrix(0, q*R, M)
for (r in 1:R) {
    h[(q*(r-1)+1):(q*(r)), ] <- t(rhalton(M, q, singleseed = r))
}
z <- qnorm(h)

test <- laplaceIS_sml2(y = manifest,
              A = constrMat,
              z = z,

              lambda0 = lambda0_init,
              lambda = lambda_init,
              transformed_rhos = transformed_rhos_init,

              maxiter = 5,
              tol = 1e-5,
              linkFLAG = 1,
              corrFLAG = 1,
              grFLAG = 1,
              ncores= 1,
              mauto_minM = 100,
              mauto_each = 100)
test



nllR <- function(par){
    Lam0.vec <- par[1:length(lambda0_init)]
    Lam.vec <- par[(length(lambda0_init)+1):(length(lambda0_init)+length(lambda_init))]
    transformed_rhos_vec <- par[(length(lambda0_init)+length(lambda_init)+1):length(par)]
    obj <- laplaceIS_sml2(y = manifest,
                         A = constrMat,
                         z = z,
                         lambda0 = Lam0.vec,
                         lambda = Lam.vec,
                         transformed_rhos = transformed_rhos_vec,
                         maxiter = 5,
                         tol = 1e-5,
                         linkFLAG = 1,
                         corrFLAG = 1,
                         grFLAG = 0,
                         ncores = 1,
                         mauto_minM = 100,
                         mauto_each = 100
    )
    out <- obj$nll
    return(out)
}

nllR(c(lambda0_init, lambda_init, transformed_rhos_init))
numDeriv::grad(nllR, c(lambda0_init, lambda_init, transformed_rhos_init))

mod <- fit_gllvmSML(
        manifest = as.matrix(manifest),
        link = 'logit',
        constraintMat = constrMat,
        corrFLAG = 0,
        rqmcFLAG = 1,
        maxiter = 5,
        newton_tol = 1e-5,
        M = 500,
        R = 1,
        silent = F,
        ncores = 4,
        traceFLAG = 0,
        mauto_minM = 500,
        mauto_each = 50,
        mauto_tol = 1e-6)



mod$loadings
mod$inner_converged_units
mod$psi
mod$time
mean(mod$var_rqmc)
mod$path_var
rhalton(10, 4)


lapply(list(mod500, modAuto, modAuto2), function(x) x$time)
as_tibble(qnorm(rhalton(n = 10000, d = 2, singleseed = 2)) )%>%
    ggplot(aes(x = V1, y = V2))+
    geom_point()

as_tibble(matrix(rnorm(500*2), 500, 2)) %>%
    ggplot(aes(x = V1, y = V2))+
    geom_point()



#############
library(mirt)
fit_mirtEM <- function(manifest, constraintMat, corrFLAG =0, met = 'EM', itemType = NULL){
    require(mirt)
    start_time <- Sys.time()
    mirt_mod <- ''
    for (lat in 1:ncol(constraintMat)) {
        plusFLAG <- 0
        mirt_mod <- paste0(mirt_mod, 'F', lat, ' = ')
        for (item in 1:nrow(constraintMat)) {
            if(constraintMat[item,lat]!=0){
                if(plusFLAG == 1){
                    mirt_mod <- paste0(mirt_mod, ',', item)
                }else{
                    mirt_mod <- paste0(mirt_mod, item)
                }
                plusFLAG <- 1
            }
        }
        mirt_mod <- paste0(mirt_mod, '\n   ')
    }
    if(corrFLAG==1){
        mirt_mod <- paste0(mirt_mod, 'COV = F1')
        for (lat in 2:ncol(constraintMat)) {
            mirt_mod <- paste0(mirt_mod, '*F', lat)
        }
    }


    fit.mirt <- mirt(as.data.frame(manifest), mirt_mod, method = met, verbose = F, quadpts = 140, technical = list(NCYCLES=5000) )
    loadingsMIRT <- as.matrix(as_tibble(coef(fit.mirt, simplify = T)$items) %>% select(starts_with('a')) )
    loadingsMIRTstd <- extract.mirt(fit.mirt, 'F')
    #thresholdsMIRT <- as_tibble(coef(fit.mirt, simplify = T)$items) %>% select(starts_with('d'))
    thresholdsMIRT <- as.matrix(as_tibble(coef(fit.mirt, simplify = T)$items) %>% select(starts_with('d')) )
    #thresholdsMIRT <- thresholdsMIRT[, ncol(thresholdsMIRT):1]
    #thresholdsMIRT <- split(thresholdsMIRT, row(thresholdsMIRT)[,1])
    names(thresholdsMIRT) <- NULL
    latent_covMIRT <- coef(fit.mirt, simplify = T)$cov
    end_time <- Sys.time()

    out <- list(
        mod_syntax = mirt_mod,
        obj = fit.mirt,
        intercepts = thresholdsMIRT,
        loadings = loadingsMIRT,
        loadings_std = loadingsMIRTstd,

        psi = latent_covMIRT,
        time = difftime(end_time, start_time, units = ("secs"))[[1]]
    )

    return(out)
}
modEM <- fit_mirtEM(manifest, constrMat, corrFLAG = 0, met = 'EM')
modEM$loadings
modEM$psi
modEM$time

#############
theta_path <- list()
iter <- 1
for (i in 1:length(modtxt)) {
    if(substr(modtxt[i], 1, 4) == ' x ='){
        theta_txt <- substr(modtxt[i], 5, nchar(modtxt[i]))
        theta_path[[iter]] <- as.numeric(strsplit(gsub(" ", "", theta_txt, fixed = TRUE), ',')[[1]])
        iter = iter + 1
    }
}

dim(modtxt[1])
substr(modtxt[1], 1, 4)
test_txt <- as.numeric(strsplit(gsub(" ", "", theta_path[[1]], fixed = TRUE), ',')[[1]])
as.numeric(test_txt[[1]])
