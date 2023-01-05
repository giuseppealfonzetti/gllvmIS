#' @export
fit_gllvmSML <- function(
  manifest,        # manifest dataset (n x p)
  link = 'logit',  # 'logit or 'probit'
  constraintMat,   # constraints for loadings matrix (p x q)
  corrFLAG = 0,    # estimate latent covariance matrix
  rqmcFLAG = T,    # use quasi randomized monte carlo for IS
  maxiter = 10,    # maximum iterations inner newton optimization problem
  newton_tol = 1e-5, # inner optimization newton tolerance
  M = 100,         # number of samples for IS
  R = 1,
  seed = 123,      # seed
  silent = F,      # silent mode
  infoFLAG = T,
  ncores = 1,
  traceFLAG = 0,
  mauto_minM = 100,
  mauto_each = 100,
  mauto_tol = 1e-5
){
  start_time <- Sys.time()
  set.seed(seed)

  #### PREPARING MODEL INPUT
  n <- nrow(manifest) ### number of subjects
  p <- ncol(manifest) ### number of items
  q <- ncol(constraintMat) ### number of latent variables

  lambda0_init = qlogis(colMeans(manifest))
  lambda_init = rep(0.5, sum(constraintMat))


  transformed_rhos_init = rep(0, q*(q-1)/2)

  if(link == 'logit'){
    link = 0
  } else if (link == 'probit'){
    link = 1
  }

  # samples for IS
  if(rqmcFLAG == T){
    if(silent == F) cat('Drawing rqmc samples for IS.\n')
    h <- matrix(0, q*R, M)
    for (r in 1:R) {

      if(r == 1){
        h[(q*(r-1)+1):(q*(r)), ] <- t(rhalton(M, q, singleseed = r))
      }else{
        h[(q*(r-1)+1):(q*(r)), ] <- t(rhalton(M, q, singleseed = r))
      }
    }
    z <- qnorm(h)
  } else {
    if(silent == F) cat('Drawing samples for IS.\n')
    z <- matrix(rnorm(q*M), q, M)
  }

  #### PREPARING FITTING FUNCTIONS

  # function for nll
  ISnll <- function(par){
    Lam0.vec <- par[1:length(lambda0_init)]
    Lam.vec <- par[(length(lambda0_init)+1):(length(lambda0_init)+length(lambda_init))]
    transformed_rhos_vec <- par[(length(lambda0_init)+length(lambda_init)+1):length(par)]
    rhos_vec = (exp(2*transformed_rhos_vec)-1)/(exp(2*transformed_rhos_vec)+1)
    obj <- laplaceIS_sml2(y = manifest,
                         A = constraintMat,
                         z = z,
                         lambda0 = Lam0.vec,
                         lambda = Lam.vec,
                         transformed_rhos = transformed_rhos_vec,
                         maxiter = maxiter,
                         tol = newton_tol,
                         linkFLAG = link,
                         corrFLAG = corrFLAG,
                         grFLAG = 0,
                         ncores = ncores,
                         mauto_minM = mauto_minM,
                         mauto_each = mauto_each,
                         mauto_tol = mauto_tol
    )
    out <- obj$nll
    return(out)
  }

  # function for gradient
  ISgr <- function(par){
    Lam0.vec <- par[1:length(lambda0_init)]
    Lam.vec <- par[(length(lambda0_init)+1):(length(lambda0_init)+length(lambda_init))]
    transformed_rhos_vec <- par[(length(lambda0_init)+length(lambda_init)+1):length(par)]
    obj <- laplaceIS_sml2(y = manifest,
                         A = constraintMat,
                         z = z,
                         lambda0 = Lam0.vec,
                         lambda = Lam.vec,
                         transformed_rhos = transformed_rhos_vec,
                         maxiter = maxiter,
                         tol = newton_tol,
                         linkFLAG = link,
                         corrFLAG = corrFLAG,
                         grFLAG = 1,
                         ncores = ncores,
                         mauto_minM = mauto_minM,
                         mauto_each = mauto_each,
                         mauto_tol = mauto_tol
    )
    out <- obj$gradient
    return(out)
  }

  # parameter vector
  par <- c(lambda0_init, lambda_init, transformed_rhos_init)
  par_init <- par
  #### FIT
  if(silent == F) cat('Optimizing the model...\n')
  trace_txt <- capture.output(
    opt <- ucminf::ucminf(par, ISnll, ISgr, control = list(trace = traceFLAG))
  )
  par <- opt$par

  #### REARRANGE RESULTS
  if(silent == F) cat('Storing results...\n')
  fit = list()
  fit$theta_init <- par_init
  fit$theta <- par
  fit$z <- z
  # Fit message
  fit$convergence <- opt$convergence

  # intercept
  intercepts <- par[1:length(lambda0_init)]
  fit$intercepts <- intercepts

  # loadings
  lambda <- par[(length(lambda0_init)+1):(length(lambda0_init)+length(lambda_init))]
  fit$loadings <- constraintMat
  s = 1
  for(j in 1:p){
    for(h in 1:q){
      #if(j==h)fit$loadings[j, h] = 1
      if(constraintMat[j, h] != 0.0)
      {
        fit$loadings[j, h] = lambda[s]
        s = s+1
      }
    }
  }


  # correlations
  trhos <- par[(length(lambda0_init)+length(lambda_init)+1):length(par)]
  rhos <- (exp(2*trhos)-1)/(exp(2*trhos)+1)

  fit$psi <- matrix(1, q, q)
  s = 1
  for( h in 1:q){
    for(j in 1:q){
      if(j > h)
      {
        fit$psi[j, h] = rhos[s]
        fit$psi[h, j] = rhos[s]
        s = s + 1
      }
    }
  }

  # nll
  fit$nll <- opt$value

  # inner optimization output
  if(infoFLAG == 1){
    obj <- laplaceIS_sml2(y = manifest,
                         A = constraintMat,
                         z = z,
                         lambda0 = intercepts,
                         lambda = lambda,
                         transformed_rhos = trhos,
                         maxiter = maxiter,
                         tol = newton_tol,
                         linkFLAG = link,
                         corrFLAG = corrFLAG,
                         grFLAG = 0,
                         ncores = ncores,
                         mauto_minM = mauto_minM,
                         mauto_each = mauto_each,
                         mauto_tol = mauto_tol
    )
    fit$scores <- obj$scores
    fit$inner_converged_units = obj$converged_units
    fit$optInfo <- opt$info
    if(R>1){
      fit$var_rqmc <- obj$var_rqmc
    }

  }

  #fit$txt <- trace_txt

  if(traceFLAG == 1){

    # function for nll
    ISvar <- function(par){
      Lam0.vec <- par[1:length(lambda0_init)]
      Lam.vec <- par[(length(lambda0_init)+1):(length(lambda0_init)+length(lambda_init))]
      transformed_rhos_vec <- par[(length(lambda0_init)+length(lambda_init)+1):length(par)]
      rhos_vec = (exp(2*transformed_rhos_vec)-1)/(exp(2*transformed_rhos_vec)+1)
      obj <- laplaceIS_sml2(y = manifest,
                            A = constraintMat,
                            z = z,
                            lambda0 = Lam0.vec,
                            lambda = Lam.vec,
                            transformed_rhos = transformed_rhos_vec,
                            maxiter = maxiter,
                            tol = newton_tol,
                            linkFLAG = link,
                            corrFLAG = corrFLAG,
                            grFLAG = 0,
                            ncores = ncores,
                            mauto_minM = mauto_minM,
                            mauto_each = mauto_each,
                            mauto_tol = mauto_tol
      )
      out <- mean(obj$var_rqmc)
      return(out)
    }

    path_theta <- list()
    iter <- 1
    for (i in 1:length(trace_txt)) {
      if(substr(trace_txt[i], 1, 4) == ' x ='){
        theta_txt <- substr(trace_txt[i], 5, nchar(trace_txt[i]))
        path_theta[[iter]] <- as.numeric(strsplit(gsub(" ", "", theta_txt, fixed = TRUE), ',')[[1]])
        iter = iter + 1
      }
    }

    fit$path_theta <- path_theta

    path_var <- c()
    for (i in 1:length(path_theta)){
      path_var[i] <- ISvar(path_theta[[i]])
    }

    fit$path_var <- path_var
  }


  # time
  end_time <- Sys.time()
  fit$time <- difftime(end_time, start_time, units = ("secs"))[[1]]
  if(silent == F) cat('Completed!\n')

  return(fit)


}

#' @export
fit_gllvmSML2 <- function(
    DATA_LIST = list('DATA', 'CONSTRMAT', 'CORRFLAG'),
    CPP_CONTROL = list(),
    UCMINF_CONTROL = list('ctrl' = list(), 'hessian' = 0 ),
    INIT = NULL,
    VERBOSEFLAG = 0,
    NCORES = 1,
    RQMCFLAG = 0,
    INFOFLAG = T
){

  out <- list()
  start_time <- Sys.time()
  # Identify model dimensions
  p <- ncol(DATA_LIST$DATA)
  n <- nrow(DATA_LIST$DATA)
  q <- ncol(DATA_LIST$CONSTRMAT)
  M <- CPP_CONTROL$M
  R <- CPP_CONTROL$R
  d = p + sum(DATA_LIST$CONSTRMAT) + q*(q-1)/2

  # Check Constraints
  if(is.null(DATA_LIST$CONSTRMAT)){
    stop('CONSTRMAT not declared')
  }else if(nrow(DATA_LIST$CONSTRMAT)!=ncol(DATA_LIST$DATA) || ncol(DATA_LIST$CONSTRMAT)>=nrow(DATA_LIST$CONSTRMAT)){

    stop('CONSTRMAT dimensions not acceptable. Check Items x Factors.')
  }
  out$constraints <- DATA_LIST$CONSTRMAT


  # Check Initialisation
  if(is.vector(INIT)){
    if(length(INIT)!=d)
      stop(paste0('init vector has length ', length(INIT), ' instead of ', d ,'.'))
    else
      message('1. Initialising at init vector.')
    out$theta_init <-  INIT
  }else{
    if(is.null(INIT))
      message('1. Initialising at default values')
    lambda0_init = qlogis(colMeans(DATA_LIST$DATA))
    lambda_init = rep(0.1, sum(DATA_LIST$CONSTRMAT))
    transformed_rhos_init = rep(0, q*(q-1)/2)
    out$theta_init <-  c(lambda0_init, lambda_init, transformed_rhos_init)
  }


  ## Check the link
  if(CPP_CONTROL$LINK == 'logit'){
    out$link = 0
  } else if (CPP_CONTROL$LINK == 'probit'){
    out$link = 1
  }

  # samples for IS
  if(RQMCFLAG == 1){
    message('2. Drawing rqmc Halton (Owen, 2017).')
    h <- matrix(0, q*R, M)
    for (r in 1:R) {

        h[(q*(r-1)+1):(q*(r)), ] <- t(rhalton(M, q, singleseed = r))

    }
    zsamples <- qnorm(h)
  } else if(RQMCFLAG == 2){
    message('2. Drawing rqmc Halton (qrng::ghalton).')
    h <- matrix(0, q*R, M)
    for (r in 1:R) {

      set.seed(r)
      h[(q*(r-1)+1):(q*(r)), ] <- t(qrng::ghalton(CPP_CONTROL$M, d = q, method = 'generalized'))

    }
    zsamples <- qnorm(h)
  } else {
    message('2. Drawing samples MC samples.')
    zsamples <- matrix(rnorm(q*M), q, M)
  }

  out$z <- zsamples

  # function for nll
  ISnll <- function(par){
    Lam0.vec <- par[1:p]
    Lam.vec <- par[(p+1):(p+sum(DATA_LIST$CONSTRMAT))]
    transformed_rhos_vec <- par[(p+sum(DATA_LIST$CONSTRMAT)+1):length(par)]
    if(!DATA_LIST$CORRFLAG) transformed_rhos_vec <- rep(0,q*(q-1)/2)
    obj <- laplaceIS_sml2(y = DATA_LIST$DATA,
                          A = DATA_LIST$CONSTRMAT,
                          z = zsamples,
                          lambda0 = Lam0.vec,
                          lambda = Lam.vec,
                          transformed_rhos = transformed_rhos_vec,
                          maxiter = CPP_CONTROL$MAXITER,
                          tol = CPP_CONTROL$NEWTON_TOL,
                          linkFLAG = out$link,
                          corrFLAG = DATA_LIST$CORRFLAG,
                          grFLAG = 0,
                          ncores = NCORES,
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
    Lam.vec <- par[(p+1):(p+sum(DATA_LIST$CONSTRMAT))]
    transformed_rhos_vec <- par[(p+sum(DATA_LIST$CONSTRMAT)+1):length(par)]
    obj <- laplaceIS_sml2(y = DATA_LIST$DATA,
                          A = DATA_LIST$CONSTRMAT,
                          z = zsamples,
                          lambda0 = Lam0.vec,
                          lambda = Lam.vec,
                          transformed_rhos = transformed_rhos_vec,
                          maxiter = CPP_CONTROL$MAXITER,
                          tol = CPP_CONTROL$NEWTON_TOL,
                          linkFLAG = out$link,
                          corrFLAG = DATA_LIST$CORRFLAG,
                          grFLAG = 1,
                          ncores = NCORES,
                          mauto_minM = M,
                          mauto_each = 1,
                          mauto_tol = 1e-6
    )
    out <- obj$gradient
    return(out)
  }

  # list of ucminf args
  args <- list(
    'par' = out$theta_init,
    'fn' = ISnll,
    'gr' = ISgr,
    'control' = UCMINF_CONTROL$ctrl,
    'hessian' = UCMINF_CONTROL$hessian)

  # optimisation
  message('3. Optimisation...')
  start_opt <- Sys.time()
  opt <- do.call(ucminf::ucminf, args)
  end_opt <- Sys.time()
  out$num_time <- as.numeric(difftime(end_opt, start_opt, units = 'secs')[1])

  out$fit <- opt

  out$control <- UCMINF_CONTROL
  out$theta   <- opt$par

  if(INFOFLAG){
    ISobj <- function(par){
      Lam0.vec <- par[1:p]
      Lam.vec <- par[(p+1):(p+sum(DATA_LIST$CONSTRMAT))]
      transformed_rhos_vec <- par[(p+sum(DATA_LIST$CONSTRMAT)+1):length(par)]
      #rhos_vec = (exp(2*transformed_rhos_vec)-1)/(exp(2*transformed_rhos_vec)+1)
      obj <- laplaceIS_sml2(y = DATA_LIST$DATA,
                            A = DATA_LIST$CONSTRMAT,
                            z = zsamples,
                            lambda0 = Lam0.vec,
                            lambda = Lam.vec,
                            transformed_rhos = transformed_rhos_vec,
                            maxiter = CPP_CONTROL$MAXITER,
                            tol = CPP_CONTROL$NEWTON_TOL,
                            linkFLAG = out$link,
                            corrFLAG = DATA_LIST$CORRFLAG,
                            grFLAG = 0,
                            ncores = 1,
                            mauto_minM = M,
                            mauto_each = 1,
                            mauto_tol = 1e-6
      )
      out <- obj
      return(out)
    }
    obj <- ISobj(opt$par)
    out$scores <- obj$scores
    out$inner_converged_units = obj$converged_units
    out$optInfo <- opt$info
    if(R>1){
      out$var_rqmc <- obj$var_rqmc
    }

  }
  end_time <- Sys.time()
  out$time <- as.numeric(difftime(end_time, start_time, units = 'secs')[1])
  message('4. Done! (', round(out$time,2),' secs)')

  return(out)
}


#' @export
fit_mirtEM <- function(manifest, constraintMat, corrFLAG = 0, met = 'EM', itemType = NULL, QUADPTS, NCYCL){
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


  fit.mirt <- mirt::mirt(as.data.frame(manifest), mirt_mod, method = met, verbose = F, quadpts = QUADPTS, technical = list(NCYCLES=NCYCL) )
  loadingsMIRT <- as.matrix(as_tibble(coef(fit.mirt, simplify = T)$items) %>% select(starts_with('a')) )
  loadingsMIRTstd <- mirt::extract.mirt(fit.mirt, 'F')
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

  # out <- list()
  # out$time = difftime(end_time, start_time, units = ("secs"))[[1]]
  # out$fit <- fit.mirt
  message('Done! (', round(out$time,2),' secs)')

  return(out)
}
