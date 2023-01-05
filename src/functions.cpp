#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppThread.h>

#include "itemConditional.h"
#include "jointNll.h"
#include "mainOld.h"

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppThread)]]



// Main function. Simulate log likelihood with importance sampling and
// compute analytical gradient with respect to loadings and correlations
//std::mutex mtx;

//' @export
// [[Rcpp::export]]
Rcpp::List laplaceIS_sml2(Eigen::Map<Eigen::MatrixXd> y,
                         Eigen::Map<Eigen::MatrixXd> A,
                         Eigen::Map<Eigen::MatrixXd> z,

                         Eigen::Map<Eigen::VectorXd> lambda0,
                         Eigen::Map<Eigen::VectorXd> lambda,
                         Eigen::Map<Eigen::VectorXd> transformed_rhos,

                         const unsigned int maxiter,
                         const double tol,
                         const unsigned int linkFLAG,
                         const bool corrFLAG,
                         const bool grFLAG,
                         const unsigned int ncores,

                         const unsigned int mauto_minM = 100,
                         const unsigned int mauto_each = 50,
                         const double mauto_tol = 1e-5){

  Eigen::VectorXd rhos = ((Eigen::VectorXd(2*transformed_rhos)).array().exp() - 1)/((Eigen::VectorXd(2*transformed_rhos)).array().exp() + 1);  // reparametrize rhos
  const unsigned int n = y.rows(); // number of units
  const unsigned int p = A.rows(); // number of items
  const unsigned int q = A.cols(); // number of latents
  const unsigned int M = z.cols(); // number of samples for IS
  const unsigned int R = z.rows()/q;
  const unsigned int d = lambda0.size() + lambda.size() + transformed_rhos.size(); // number of parameters
  unsigned int iter; //iterator
  unsigned int converged_units = 0; // counts units converged according to tol during inner optimization

  // Identity matrix q x q
  const Eigen::MatrixXd Iq = Eigen::MatrixXd::Identity(q,q);

  // Initialize latent scores matrix
  Eigen::MatrixXd u = Eigen::MatrixXd::Zero(n,q);

  // latent variable covariance matrix
  Eigen::MatrixXd Sigma = Eigen::MatrixXd::Ones(q,q);
  iter = 0;
  for(unsigned int j = 0; j < q; j++){
    for(unsigned int h = 0; h < j; h++){
        Sigma(j, h) = Sigma(h, j) = rhos(iter);
        iter ++;
    }
  }

  // Latent covariance ldlt solver object
  Eigen::LDLT<Eigen::MatrixXd> Sigma_ldlt(Sigma);
  const double Sigma_logdet = Sigma_ldlt.vectorD().array().abs().log().sum();
  const Eigen::MatrixXd Sigma_inv = Sigma_ldlt.solve(Iq);

  // initialize output of parallel_task
  double nll = 0;
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(d);
  //Eigen::VectorXd var_rqmc = Eigen::VectorXd::Ones(n);
  std::vector<double> var_rqmc(n);
  std::vector<unsigned int> chosen_M(n);

  // Spawning threads
  //RcppThread::ThreadPool pool(ncores);

  // Lambda function to be computed in parallel
  auto parallel_task = [&y, &u, &z, &n, &p, &q, &M, &R, &d, &A, &lambda0, &lambda, &transformed_rhos, &Sigma_inv, &Sigma_logdet, &linkFLAG, &corrFLAG, &grFLAG, &maxiter, &tol, &Iq, &converged_units, &nll, &gradient, &var_rqmc, &mauto_minM, &mauto_each, &mauto_tol, &chosen_M](unsigned int i){

    // data and latent vars of unit i
    const Eigen::VectorXd yi = y.row(i);
    Eigen::VectorXd ui = u.row(i);


    // auto id = std::this_thread::get_id();
    // RcppThread::Rcout << "id: " << id << " , i: " << i << "\n" << "yi: \n" << yi << "\n" <<  "ui: \n" << ui << "\n" << "zi: \n" << zi << std::endl;


    /* LAPLACE OPTIMIZATION */
    // negative hessian at the optimum for unit i
    Eigen::MatrixXd Hi;

    // setup response type
    ef_item item; item.linkFLAG = linkFLAG;

    // setup joint negative log likelihood
    joint_nll jnlli;
    jnlli.item = item;
    jnlli.A = A;
    jnlli.set_par_(lambda0, lambda, Sigma_inv, Sigma_logdet);
    jnlli.update_y_(yi);

    // optimization
    NewtonOpt opt;
    opt.setup_(jnlli, maxiter, tol);
    opt.optimize_(ui, Hi);
    u.row(i) = ui;

    /* QUANTITIES FOR SIMULATED LIKELIHOOD*/
    // compute jnll evaluated at uopt
    jnlli.update_u_(ui);
    const double jll_uopt = -jnlli.jnll_();

    // Compute H^{-1}
    Eigen::LDLT<Eigen::MatrixXd> ldlt(Hi);
    const Eigen::MatrixXd Hinvi = ldlt.solve(Iq);

    // Compute Cholesky decomposition of H^{-1}
    Eigen::LLT<Eigen::MatrixXd> llt(Hinvi);
    Eigen::MatrixXd Ctmp = llt.matrixL();
    if(Ctmp(1,1)<0) Ctmp = -Ctmp; // identify cholesky with positive diagonal
    const Eigen::SparseMatrix<double> Ci = Ctmp.sparseView();

    Eigen::MatrixXd tw = Eigen::MatrixXd::Zero(R,M);
    double tw_av;
    double var_tw_av;
    //Rcpp::Rcout << "i: " << i << "--> ";
    for(unsigned int m = 0; m < M; m++){
      for(unsigned int r = 0; r < R; r++){
        const Eigen::VectorXd z_mr = z.block(q*(r), m, q, 1);
        const Eigen::VectorXd v_mr = ui + Ci*z_mr; // N(u, H^{-1})

        jnlli.update_u_(v_mr);
        tw(r, m) = exp(-jnlli.jnll_() - jll_uopt + .5 * z_mr.dot(z_mr));

      }


      // if((m >= (mauto_minM-1)) & ((m+1)%mauto_each == 0)){
      //   Eigen::MatrixXd tmp_tw = tw.block(0,0,R,m+1);
      //   Eigen::VectorXd tmp_tw_av_M = tmp_tw.rowwise().mean();
      //   const double tmp_tw_av = tmp_tw_av_M.mean();
      //   double tmp_var_tw_av = (tmp_tw_av_M.array()-tmp_tw_av).square().sum()/(R*(R-1)); //(1/(R*(R-1)))*
      //
      //   if(R==1) tmp_var_tw_av = 0;
      //   //Rcpp::Rcout << " | m: " << m + 1 << ", var = " << tmp_var_tw_av;
      //   if(tmp_var_tw_av < mauto_tol){
      //     chosen_M[i] = m+1;
      //     tw_av = tmp_tw_av;
      //     var_tw_av = tmp_var_tw_av;
      //     break;
      //   }
      // }
    }
    //Rcpp::Rcout << "\n";

    const Eigen::VectorXd tw_av_M = tw.rowwise().mean();  // average over M samples
    tw_av = tw_av_M.mean(); // average of M-averages over R sequences
    var_tw_av = (tw_av_M.array()-tw_av).square().sum()/(R*(R-1)); //(1/(R*(R-1)))*

    //Rcpp::Rcout << "i: " << i << ", mean: " << tw_av <<  ", var: "<< var_tw_av << "\n";



    /*QUANTITIES RELATED TO GRADIENT COMPUTATION*/
    Eigen::VectorXd g_i = Eigen::VectorXd::Zero(d);
    if(grFLAG){
      jnlli.update_u_(ui);
      unsigned int iter = 0;

        // INTERCEPTS
        for(unsigned int j = 0; j < p; j++){

          // individual partial derivatives
          jnlli.Puopt_Plamj0_(Hinvi, j);
          jnlli.PC_Plamj0_(Ci, Hinvi, j);
          g_i(iter) += jnlli.PlogdetC_Plamj0_(Ci, Hinvi, j);

          // individual partial derivatives along the MC sample
          double tmp = 0;
          for(unsigned int r = 0; r < R; r++){
            // for(unsigned int m = 0; m < chosen_M[i]; m++){
            for(unsigned int m = 0; m < M; m++){
              const Eigen::VectorXd z_mr = z.block(q*(r), m, q, 1);
              const double w_mr = tw(r, m);
              tmp += w_mr * jnlli.Pjllv_Plamj0_(Ci, Hinvi, j, z_mr);
            }
          }

          // g_i(iter) += (1/(chosen_M[i]*R*tw_av))*tmp;
          g_i(iter) += (1/(M*R*tw_av))*tmp;
          iter ++;
        }

        // LOADINGS
        for(unsigned int j = 0; j < p; j++){
          for(unsigned int s = 0; s < q; s++){
            if(A(j,s)!=0){

              // individual partial derivatives
              jnlli.Puopt_Plamjs_(Hinvi, j, s);
              jnlli.PC_Plamjs_(Ci, Hinvi, j, s);
              g_i(iter) += jnlli.PlogdetC_Plamjs_(Ci, Hinvi, j, s);

              // individual partial derivatives along the MC sample
              double tmp = 0;
              for(unsigned int r = 0; r < R; r++){
                // for(unsigned int m = 0; m < chosen_M[i]; m++){
                for(unsigned int m = 0; m < M; m++){
                  const Eigen::VectorXd z_mr = z.block(q*(r), m, q, 1);
                  const double w_mr = tw(r, m);
                  tmp += w_mr * jnlli.Pjllv_Plamjs_(Ci, Hinvi, j, s, z_mr);

                }
              }

              // g_i(iter) += (1/(chosen_M[i]*R*tw_av))*tmp;
              g_i(iter) += (1/(M*R*tw_av))*tmp;
              iter ++;

            }
          }
        }

        // CORRELATIONS
        unsigned int iter1 = 0; //identify element correlations' vector parameter
        if(corrFLAG){
          for(unsigned int l = 1; l<q; l++){
            for(unsigned int s = 0; s<l; s++){

              // individual partial derivatives
              jnlli.PSigma_Ptrhors_(l, s, transformed_rhos(iter1));
              jnlli.Puopt_Ptrhors_(Hinvi, l, s, transformed_rhos(iter1));
              jnlli.PC_Ptrhors_(Ci, Hinvi, l, s, transformed_rhos(iter1));
              g_i(iter) += jnlli.PlogdetC_Ptrhors_(Ci, Hinvi, l, s, transformed_rhos(iter1));

              // individual partial derivatives along the MC sample
              double tmp = 0;
              for(unsigned int r = 0; r < R; r++){
                // for(unsigned int m = 0; m < chosen_M[i]; m++){
                for(unsigned int m = 0; m < M; m++){
                  const Eigen::VectorXd z_mr = z.block(q*(r), m, q, 1);
                  const double w_mr = tw(r, m);
                  tmp += w_mr * jnlli.Pjllv_Ptrhors_(Ci, Hinvi, l, s, transformed_rhos(iter1), z_mr);
                }
              }

              // g_i(iter) += (1/(chosen_M[i]*R*tw_av))*tmp;
              g_i(iter) += (1/(M*R*tw_av))*tmp;
              iter1 ++;
              iter ++;

            }
          }
        }


    }


    // Update section: locked to be accessed from one thread at a time
    //mtx.lock();
    if(opt.last_tol < tol) converged_units ++;
    nll -= log(Ci.diagonal().prod()) + jll_uopt + log(tw_av);
    gradient -= g_i;
    var_rqmc[i] = var_tw_av;
    //mtx.unlock();

    };

  // Compute parallel task for each unit passing it to the thread pool
  for(unsigned int i = 0; i < n; i++){
    //pool.push(parallel_task, i); // parallel
    parallel_task(i); // non parallel
  }
  //pool.join();


  Rcpp::List output =
    Rcpp::List::create(
      Rcpp::Named("nll") = nll,
      Rcpp::Named("converged_units") = converged_units,
      Rcpp::Named("gradient") = gradient,
      Rcpp::Named("scores") = u,
      Rcpp::Named("var_rqmc") = var_rqmc,
      Rcpp::Named("chosen_M") = chosen_M

    );

  return(output);
}

