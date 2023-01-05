#### functions ####
# build loadings constraint matrix [pxq]
#' @export
build_constrMat <- function(p, q, struct = 'triangular'){
  if(struct == 'triangular'){
    # Build lower trinagular matrix
    constrMat <- matrix(1, nrow = p, ncol = q)
    for(j in 1:p){
      for (h in 1:q) {
        if(h > j) constrMat[j,h] <- 0
      }
    }

  } else if ( struct == 'simple'){
    # Each item to one factor
    loadings_per_factor <- p %/% q
    remaining_loadings <- p %% q

    constrMat <- matrix(0, nrow = p, ncol = q)
    for (h in 1:(q-1)) {
      constrMat[ (loadings_per_factor*(h-1)+1) : (loadings_per_factor*(h-1) + loadings_per_factor), h] <- rep(1,loadings_per_factor)
    }
    h <- q
    constrMat[ (loadings_per_factor*(h-1)+1) : (loadings_per_factor*(h-1) + loadings_per_factor + remaining_loadings), h] <- rep(1,loadings_per_factor + remaining_loadings)

  }
  return(constrMat)
}

# generate noramal random latent scores [nxq]
#' @export
gen_scores <- function(n, q, rho = 0, seed = 123){
  set.seed(seed)
  psi <- matrix(rho, nrow = q, ncol = q)
  diag(psi) <- 1
  scores <-  mvtnorm::rmvnorm(n, mean = rep(0, nrow(psi)), sigma = psi)
  return(scores)
}

# generate the matrix of loadings [pxq] drawing from normal rv
#' @export
gen_loadings <- function(p, q, fixed = NULL, constraint_mat = NULL, seed = 123){
  set.seed(seed)
  if(is.null(constraint_mat)){
    loadings_mat <- matrix(runif(p*q, -1, 1), p, q)
    gdata::upperTriangle(loadings_mat) <-0
  } else{
    loadings_mat <-  constraint_mat
    for (j in 1:p) {
      for (h in 1:q) {
        if(loadings_mat[j,h] != 0){
          loadings_mat[j,h] <- runif(1, 0, 1)
        }
      }
    }
  }
  if(is.null(fixed)==FALSE){
    for (j in 1:p) {
      for (h in 1:q) {
        if(loadings_mat[j,h] != 0){
          loadings_mat[j,h] <- fixed
        }
      }
    }
  }
  return(loadings_mat)
}

#'@export
rmvn <- function(SAMPLE_SIZE, VAR){
  dim <- ncol(VAR)
  sample <- t(t(chol(VAR))%*%matrix(rnorm(dim*SAMPLE_SIZE), dim, SAMPLE_SIZE))
  #sample <- t(matrixprod(t(chol(VAR)), matrix(rnorm(dim*SAMPLE_SIZE), dim, SAMPLE_SIZE)))

  return(sample)
}
# generate the intercept vector
#' @export
gen_intercepts <- function(p, fixed = NULL, response = 'binary', categories = NULL, seed = 123){
  set.seed(seed)

  if(is.null(fixed)){
    # draw random intercepts
    if(response == 'binary'){
      # draw one intercept for each item
      intercepts <- matrix(runif(p, -1, 1), p, 1)
    } else if( response == 'ordinal'){
      # draw cj-1 intercepts for each item j
      intercepts <- c() ; s <- 1;
      for (j in 1:length(categories)) {
        cj = categories[j]
        for(sj in 1:(cj-1)){
          if(sj == 1){
            intercepts[s] = runif(1)
          } else{
            intercepts[s] = intercepts[s - 1] + runif(1)
          }
          s <- s + 1
        }
      }
      intercepts <- matrix(intercepts, sum(categories)-p, 1)
    }
  } else {
    # build intercept vector with given values
    if(response == 'binary'){
      intercepts <- matrix(rep(fixed, p), p, 1)
    }else if( response == 'ordinal'){
      intercepts <- c() ; s <- 0
      for (j in 1:length(categories)) {
        intercepts[(s + 1) : (s + categories[j] - 1)] <- (0 : (categories[j]-2))
        s <- s + categories[j] - 1
      }
      intercepts <- matrix(intercepts, sum(categories)-p, 1)
    }
  }
  return(intercepts)
}

# generate data
#' @export
draw_data <- function(loadings,
                      scores,
                      intercepts,
                      response = 'binary',
                      categories = NULL,
                      fam = binomial(link = 'logit'),
                      seed = 123){
    set.seed(seed)
  p <- nrow(loadings)
  q <- ncol(loadings)
  n <- nrow(scores)

  if(response == 'binary'){
    # compute eta
    eta <- t(tcrossprod(intercepts, matrix(1, n, 1)) + tcrossprod(loadings, scores))

    # check
    cond <- TRUE
    while (cond == TRUE) {
      manifest <- purrr::modify(fam$linkinv(eta), ~rbinom(1, 1,.x))
      cond <- TRUE %in% (colSums(manifest) %in% c(0, nrow(manifest)))
    }
  } else if ( response == 'ordinal'){
    # Expand the loadings matrix to be [c-p, q]
    Lam = matrix(0,sum(categories)-p, q)
    s <- 1
    for(j in 1:p){
      cj = categories[j]
      rowj = loadings[j,]
      for(sj in 1:(cj-1)){
        Lam[s,] = rowj
        s = s + 1
      }
    }

    # compute eta
    eta <- t(tcrossprod(intercepts, matrix(1, n, 1)) - tcrossprod(Lam, scores))

    # compute cumulative probabilities
    cumprob <- fam$linkinv(eta)


    margprob <- matrix(0, nrow(eta), sum(categories))
    manifest <- matrix(0, nrow(eta), length(categories))

    for (i in 1:nrow(eta)){
      # for each unit compute marginals for each category
      # of each item
      s <- 0
      for (j in 1:p) {

        # which columns
        col <- (s + 1) : (s + categories[j] - 1)

        # cj dimensional cumulative vector
        cum <- c(cumprob[i, col], 1)

        # cj marginals
        marg <- cum - c(0, cumprob[i, col] )

        # draw a response
        resp <- rmultinom(1, 1, marg)

        # store the category chosen (base level = 0)
        manifest[i,j] <- which(resp == 1) - 1
        s <- s + cat[j] - 1
      }
    }
  }

  return(manifest)

}

#' @export
generate_data <- function(
    SAMPLE_SIZE,
    LOADINGS,
    INTERCEPTS,
    LATENT_COV,
    RESPONSE = 'binary',
    CATEGORIES = NULL,
    FAM = binomial(link = 'logit'),
    SEED = 123){
  set.seed(SEED)
  factors <- rmvn(SAMPLE_SIZE = SAMPLE_SIZE, VAR = LATENT_COV)

  p <- nrow(LOADINGS)
  q <- ncol(LOADINGS)

  if(RESPONSE == 'binary'){
    # compute eta
    eta <- t(tcrossprod(INTERCEPTS, matrix(1, SAMPLE_SIZE, 1)) + tcrossprod(LOADINGS, factors))

    # check
    cond <- TRUE
    while (cond == TRUE) {
      set.seed(SEED)
      manifest <- purrr::modify(FAM$linkinv(eta), ~rbinom(1, 1,.x))
      cond <- TRUE %in% (colSums(manifest) %in% c(0, nrow(manifest)))
    }
  } else if ( RESPONSE == 'ordinal'){
    # Expand the loadings matrix to be [c-p, q]
    Lam = matrix(0,sum(categories)-p, q)
    s <- 1
    for(j in 1:p){
      cj = categories[j]
      rowj = loadings[j,]
      for(sj in 1:(cj-1)){
        Lam[s,] = rowj
        s = s + 1
      }
    }

    # compute eta
    eta <- t(tcrossprod(INTERCEPTS, matrix(1, n, 1)) - tcrossprod(Lam, factors))

    # compute cumulative probabilities
    cumprob <- FAM$linkinv(eta)


    margprob <- matrix(0, nrow(eta), sum(categories))
    manifest <- matrix(0, nrow(eta), length(categories))

    for (i in 1:nrow(eta)){
      # for each unit compute marginals for each category
      # of each item
      s <- 0
      for (j in 1:p) {

        # which columns
        col <- (s + 1) : (s + categories[j] - 1)

        # cj dimensional cumulative vector
        cum <- c(cumprob[i, col], 1)

        # cj marginals
        marg <- cum - c(0, cumprob[i, col] )

        # draw a response
        resp <- rmultinom(1, 1, marg)

        # store the category chosen (base level = 0)
        manifest[i,j] <- which(resp == 1) - 1
        s <- s + cat[j] - 1
      }
    }
  }

  return(manifest)

}





















