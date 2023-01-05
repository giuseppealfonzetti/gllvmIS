#ifndef mainOld_H
#define mainOld_H

std::mutex mtxOld;

//' @export
// [[Rcpp::export]]
Rcpp::List laplaceIS_sml(Eigen::Map<Eigen::MatrixXd> y,
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
                         const unsigned int ncores){

    Eigen::VectorXd rhos = ((Eigen::VectorXd(2*transformed_rhos)).array().exp() - 1)/((Eigen::VectorXd(2*transformed_rhos)).array().exp() + 1);  // reparametrize rhos
    const unsigned int n = y.rows(); // number of units
    const unsigned int p = A.rows(); // number of items
    const unsigned int q = A.cols(); // number of latents
    const unsigned int M = z.cols(); // number of samples for IS
    const unsigned int d = lambda0.size() + lambda.size() + transformed_rhos.size(); // number of parameters
    unsigned int iter; //iterator
    unsigned int converged_units = 0; // counts units converged according to tol during inner optimization

    // Identity matrix q x q
    const Eigen::MatrixXd Iq = Eigen::MatrixXd::Identity(q,q);

    // Initialize latent scores matrix
    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(n,q);

    // Initialize matrix with quantities needed for sum in IS likelihood
    Eigen::VectorXd alpha = Eigen::VectorXd::Zero(M);

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
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(d,M);
    Eigen::VectorXd dlogdetC = Eigen::VectorXd::Zero(d);

    // Spawning threads
    RcppThread::ThreadPool pool(ncores);

    // Lambda function to be computed in parallel
    auto parallel_task = [&y, &u, &z, &n, &p, &q, &M, &d, &A, &lambda0, &lambda, &transformed_rhos, &Sigma_inv, &Sigma_logdet, &linkFLAG, &corrFLAG, &grFLAG, &maxiter, &tol, &Iq, &converged_units, &nll, &gradient, &alpha, &G, &dlogdetC](unsigned int i){

        // data and latent vars of unit i
        const Eigen::VectorXd yi = y.row(i);
        Eigen::VectorXd ui = u.row(i);
        Eigen::MatrixXd zi(q,M);
        for(unsigned int l = 0; l < q; l++){zi.row(l) = z.row(l*n + i);}

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

        // Compute H^(-1)
        Eigen::LDLT<Eigen::MatrixXd> ldlt(Hi);
        const Eigen::MatrixXd Hinvi = ldlt.solve(Iq);

        // Compute Cholesky decomposition of H^(-1)
        Eigen::LLT<Eigen::MatrixXd> llt(Hinvi);
        Eigen::MatrixXd Ctmp = llt.matrixL();
        if(Ctmp(1,1)<0) Ctmp = -Ctmp; // identify cholesky with positive diagonal
        const Eigen::SparseMatrix<double> Ci = Ctmp.sparseView();

        // vector with monte carlo samples for unit i
        Eigen::VectorXd alphai = Eigen::VectorXd::Zero(M);

        for(unsigned int m = 0; m < M; m++){
            // compute am_i
            const Eigen::VectorXd zm = zi.col(m);
            const Eigen::VectorXd vm = ui + Ci*zm; // N(u, H^(-1))
            jnlli.update_u_(vm);
            alphai(m) =  -jnlli.jnll_() - jll_uopt + .5 * zm.dot(zm);
        }

        /*QUANTITES RELATED TO GRADIENT COMPUTATION*/
        Eigen::VectorXd dlogdetCi = Eigen::VectorXd::Zero(d);
        Eigen::MatrixXd Gi = Eigen::MatrixXd::Zero(d,M);
        if(grFLAG){
            jnlli.update_u_(ui);

            //iterator to identify complete vector parameter element
            int iter = 0;


            // INTERCEPTS
            for(int j = 0; j < p; j++){
                jnlli.Puopt_Plamj0_(Hinvi, j);
                jnlli.PC_Plamj0_(Ci, Hinvi, j);
                dlogdetCi(iter) = jnlli.PlogdetC_Plamj0_(Ci, Hinvi, j);

                for(int m = 0; m < M; m++){
                    Eigen::VectorXd zm = zi.col(m);
                    Gi(iter, m) = jnlli.Pjllv_Plamj0_(Ci, Hinvi, j, zm);
                }
                iter ++;
            }

            // LOADINGS
            for(int j = 0; j < p; j++){
                for(int s = 0; s < q; s++){
                    if(A(j,s)!=0){
                        jnlli.Puopt_Plamjs_(Hinvi, j, s);
                        jnlli.PC_Plamjs_(Ci, Hinvi, j, s);
                        dlogdetCi(iter) = jnlli.PlogdetC_Plamjs_(Ci, Hinvi, j, s);
                        for(int m = 0; m < M; m++){
                            Eigen::VectorXd zm = zi.col(m);
                            Gi(iter, m) = jnlli.Pjllv_Plamjs_(Ci, Hinvi, j, s, zm);
                        }
                        iter ++;
                    }
                }
            }


            // CORRELATIONS
            int iter1 = 0; //identify element correlations' vector parameter
            if(corrFLAG){
                for(int r = 0; r<q; r++){
                    for(int s = 0; s<q; s++){
                        if(r>s){
                            jnlli.PSigma_Ptrhors_(r, s, transformed_rhos(iter1));
                            jnlli.Puopt_Ptrhors_(Hinvi, r, s, transformed_rhos(iter1));
                            jnlli.PC_Ptrhors_(Ci, Hinvi, r, s, transformed_rhos(iter1));
                            dlogdetCi(iter) = jnlli.PlogdetC_Ptrhors_(Ci, Hinvi, r, s, transformed_rhos(iter1));

                            for(int m = 0; m < M; m++){
                                Eigen::VectorXd zm = zi.col(m);
                                Gi(iter, m) = jnlli.Pjllv_Ptrhors_(Ci, Hinvi, r, s, transformed_rhos(iter1), zm);
                            }
                            iter1 ++;
                            iter ++;
                        }
                    }
                }
            }
        }

        // Update section: locked to be accessed from one thread at time
        mtxOld.lock();
        if(opt.last_tol < tol) converged_units ++;
        nll -= log(Ci.diagonal().prod()) + jll_uopt;
        alpha += alphai;
        dlogdetC += dlogdetCi;
        G += Gi;
        mtxOld.unlock();

    };

    // Compute parallel task for each unit passing it to the thread pool
    for(int i = 0; i < n; i++){
        pool.push(parallel_task, i);
    }
    pool.join();

    // Reduce loop outputs to compute nll and gradient
    Eigen::VectorXd expalpha = alpha.array().exp();
    nll -= log(expalpha.sum()/M);
    Eigen::VectorXd oned(d); oned.fill(1);
    Eigen::MatrixXd tmp = (oned * expalpha.transpose()).array() * G.array();
    gradient = -dlogdetC - (tmp.rowwise().sum() / expalpha.sum());



    Rcpp::List output =
        Rcpp::List::create(
            Rcpp::Named("nll") = nll,
            Rcpp::Named("converged_units") = converged_units,
            Rcpp::Named("gradient") = gradient,
            Rcpp::Named("scores") = u

        );

    return(output);
}



#endif
