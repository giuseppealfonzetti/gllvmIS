library(tidyverse)
#### Setup ###
p <- 8
q <- 2
n <- 100
##### BINARY ####
#### Synthetic data ####
resp <- 'binary'
constrMat <- build_constrMat(p = p, q = q, struct = 'simple') # 'simple' or 'triangular'
load_test <- gen_loadings(p = p, q = q, fixed = .7, constraint_mat = constrMat)
scores_test <- gen_scores(n = n, q = q, rho = 0)
intercepts_test <- gen_intercepts(p = p, fixed = 0, response = resp)
manifest <- draw_data(loadings = load_test,
                      scores = scores_test,
                      intercepts = intercepts_test,
                      response = resp,
                      seed = 1)
manifest

lambda0_init = qlogis(colMeans(manifest))
lambda_init = rep(1, sum(constrMat))
transformed_rhos_init = rep(0, q*(q-1)/2)
rhos = (exp(2*transformed_rhos_init)-1)/(exp(2*transformed_rhos_init)+1)
Sigma <- diag(1,q,q)
s = 1;
for(j in 1:q){
  for(h in 1:q){
    if(j > h)
    {
      Sigma[j, h] = rhos[s];
      Sigma[h, j] = rhos[s];
      s = s + 1;
    };
  };
}
Sigma
Sigma_inv = solve(Sigma)
Sigma_logdet = log(det(Sigma))
v_mat <- matrix(rnorm(n*q*M), n*q, M)
r <- 2
v_mat <- lapply(1:r, function(x){
  qnorm(reduce(
    lapply(1:n, function(y) t(rhalton(M, q, singleseed = y*x))),
    rbind
  ))
})

v_mat <- lapply(1:r, function(x) matrix(rnorm(n*q*M), n*q, M))
