#'@export
get_corr <- function(THETA, CONSTRMAT, FISHERFLAG = F){
    q <- ncol(CONSTRMAT)
    d <- length(THETA)
    ncorr <- q*(q-1)/2
    trho <- THETA[(d-ncorr+1):d]
    if(FISHERFLAG == F){
        out <- (exp(2*trho)-1)/(exp(2*trho)+1)
    }else if (FISHERFLAG==T){
        out <- trho
    }
    return(out)
}

#'@export
get_lambda <- function(THETA, CAT, CONSTRMAT){
    q <- ncol(CONSTRMAT)
    p <- nrow(CONSTRMAT)
    d <- length(THETA)
    c <- sum(CAT)

    ncorr <- q*(q-1)/2
    lambda <- THETA[(c-p+1):(d-ncorr)]
    lambda
}

#'@export
get_theta <- function(TAU, LOADINGS, LATENT_COV, CAT, A, TAUREPFLAG = 1){
    # thresholds params
    thr_vec <- c()
    if(TAUREPFLAG==1){
        item = 1; thr = 0
        for (i in 1:length(TAU)) {
            maxitem = CAT[item]-1
            if(thr==0){
                thr_vec[i] = TAU[i]
            }else{
                thr_vec[i] = log(TAU[i] - TAU[i-1])
            }

            if(thr==maxitem-1){
                item = item + 1
                thr = 0
            }else{
                thr = thr + 1
            }

        }
    }else if(TAUREPFLAG==0){
        thr_vec <- TAU
    }


    load_vec <- c()
    s <- 1
    for (j in 1:ncol(LOADINGS)) {
        for (i in 1:nrow(LOADINGS)) {
            if(A[i,j]!=0){
                load_vec[s] <- LOADINGS[i,j]
                s = s+1
            }
        }
    }

    corr_vec <- c()
    s <- 1
    for (i in 1:nrow(LATENT_COV)){
        for(j in 1:ncol(LATENT_COV)){
            if(i>j){
                rho <- LATENT_COV[i,j]
                corr_vec[s] <- dplyr::if_else(abs(rho)<=1,.5*log((rho+1)/(1-rho)), 0)
                s = s+1
            }
        }
    }
    theta <- c(thr_vec, load_vec, corr_vec)
    return(theta)
}
