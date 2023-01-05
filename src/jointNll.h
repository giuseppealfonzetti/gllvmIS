#ifndef jointNll_H
#define jointNll_H

// class for computations related to joint log likelihood
class joint_nll{
public:

    // class inputs
    ef_item item;
    Eigen::VectorXd y;
    Eigen::VectorXd u;
    Eigen::MatrixXd A;
    Eigen::VectorXd lambda0;
    Eigen::VectorXd lambda;
    Eigen::MatrixXd Sigma_inv;
    double Sigma_logdet;

    // class outputs
    Eigen::VectorXd gr_u;
    Eigen::MatrixXd h_u;


    // class intermediate quantities
    Eigen::MatrixXd Lam;
    Eigen::VectorXd eta;
    int p;
    int q;

    void set_par_(
            Eigen::VectorXd lambda0_input,
            Eigen::VectorXd lambda_input,
            Eigen::MatrixXd Sigma_inv_input,
            double Sigma_logdet_input){
        lambda0 = lambda0_input;
        lambda = lambda_input;
        Sigma_inv = Sigma_inv_input;
        Sigma_logdet = Sigma_logdet_input;

        p = A.rows(); // number of items
        q = A.cols(); // number of latents

        // Build the loadings matrix and set zero-constraints
        Lam = A;
        int s = 0;
        for(int h = 0; h < q; h++){
            for(int j = 0; j < p; j++){
                //if(j==h)Lam(j,h) = 1;
                if(A(j, h) != 0.0)
                {
                    Lam(j, h) = lambda(s) ;
                    s ++;
                };
            };
        }


    }

    void update_y_(Eigen::VectorXd y_input){
        y.resize(p);
        y.fill(0.0);
        y = y_input;
    }
    void update_u_(Eigen::VectorXd u_input){
        u.resize(q);
        u.fill(0.0);
        u = u_input;
    }

    // member function to compute jnll
    double jnll_(){
        double nll = 0;
        // latent contribution (multivariate gaussian)
        nll = .5 * Sigma_logdet + .5 * u.transpose() * Sigma_inv * u;

        for(int j = 0; j < p; j++){
            double yij = y(j);
            item.update_eta_(lambda0(j), Lam.row(j), u);
            item.update_nll_();
            nll += item.nll_(yij);
        }

        return nll;
    }

    void der_u_(){

        gr_u = Sigma_inv * u;
        h_u = Sigma_inv;
        for(int j = 0; j < p; j++){
            item.update_eta_(lambda0(j), Lam.row(j), u);
            item.update_nll_();
            item.update_in_opt_();
            Eigen::VectorXd grj = (1/item.phi)*(y(j)-item.b1)*item.Ptheta_Pu_();
            gr_u -= grj;
            Eigen::MatrixXd hj = (1/item.phi)*((-item.b2)*item.Ptheta_Pu_()*(item.Ptheta_Pu_()).transpose()+(y(j)-item.b1)*item.Ptheta_PuPuT_());
            h_u -= hj;

        }
    }

    Eigen::VectorXd Puopt_Plamjs;
    /* Outer optimization wrt lamjs */
    Eigen::VectorXd Puopt_Plamjs_(Eigen::MatrixXd Hinv, int j, int s){
        Puopt_Plamjs.resize(q);

        Eigen::VectorXd lamj = Lam.row(j);
        item.update_eta_(lambda0(j), lamj, u);
        item.update_out_opt_();
        Puopt_Plamjs = Hinv * (item.phi*(-item.b2*item.Ptheta_Plambdajs_(s)*item.Ptheta_Pu_()+(y(j)-item.b1)*item.Ptheta_PuPlambdajs_(s)));

        return Puopt_Plamjs;
    }

    Eigen::MatrixXd PH_Plamjs_(Eigen::MatrixXd Hinv, int j, int s){

        Eigen::MatrixXd jac(q,q); jac.fill(0.0);
        for(int r = 0; r < p; r++){

            // contribution of item r
            Eigen::MatrixXd Pr(q,q); Pr.fill(0.0);

            // update item object
            Eigen::VectorXd lamr = Lam.row(r);
            item.update_eta_(lambda0(r), lamr, u);
            item.update_out_opt_();

            // helper quantities
            Eigen::VectorXd dlamr(q); dlamr.fill(0.0);
            Eigen::MatrixXd dlamrlamrT(q,q); dlamrlamrT.fill(0.0);
            if(j==r){
                dlamr(s) = 1.0;
                for(int k = 0; k<q; k++){
                    for(int l = 0; l<q; l++){
                        if(s==k && s!=l){
                            dlamrlamrT(k,l) = lamr(l);
                        }
                        if(s==l && s!=k){
                            dlamrlamrT(k,l) = lamr(k);
                        }
                        if(s==l && s==k){
                            dlamrlamrT(k,l) = 2*lamr(s);
                        }
                    }
                }
            }



            double deta = 0; deta += dlamr.transpose()*u; deta += lamr.transpose()*Puopt_Plamjs;
            double dtheta = item.br1 * deta;
            Eigen::MatrixXd dthetauthetauT = 2*item.br1*item.br2*deta*lamr*lamr.transpose()+pow(item.br1,2)*dlamrlamrT;
            Eigen::MatrixXd dthetauuT = item.br3 * deta * lamr*lamr.transpose() + item.br2*dlamrlamrT;
            Pr += -item.b3*dtheta*item.Ptheta_Pu_()*item.Ptheta_Pu_().transpose();
            Pr += -item.b2*dthetauthetauT;
            Pr += -item.b2*dtheta*item.Ptheta_PuPuT_();
            Pr += (y(r)-item.b1)*dthetauuT;

            jac -= item.phi * Pr;
        }
        return jac;

    }

    Eigen::MatrixXd PC_Plamjs;
    Eigen::MatrixXd PC_Plamjs_(Eigen::SparseMatrix<double> C, Eigen::MatrixXd Hinv, int j, int s){
        PC_Plamjs.resize(q,q); PC_Plamjs.fill(0.0);
        Eigen::MatrixXd B(q,q); B.fill(0.0);
        for(int k = 0; k<q; k++){
            for(int l = 0; l<q; l++){
                if(k>l)B(k,l)  = 1;
                if(k<l)B(k,l)  = 0;
                if(k==l)B(k,l) =.5;
            }
        }

        Eigen::MatrixXd tmp1 = C.transpose()*PH_Plamjs_(Hinv, j, s)*C;
        Eigen::MatrixXd tmp2 = B.array()*tmp1.array();
        PC_Plamjs = -C*tmp2;
        return PC_Plamjs;
    }

    double PlogdetC_Plamjs_(Eigen::SparseMatrix<double> C, Eigen::MatrixXd Hinv, int j, int s){
        Eigen::MatrixXd Iq; Iq.setIdentity(q, q);
        Eigen::MatrixXd Cinv = C.triangularView<Eigen::Lower>().solve(Iq);
        Eigen::MatrixXd tmp1 = Cinv*PC_Plamjs;
        double g = tmp1.trace();
        return g;
    }

    double Pjllv_Plamjs_(Eigen::SparseMatrix<double> C, Eigen::MatrixXd Hinv, int j, int s, Eigen::VectorXd z){
        Eigen::VectorXd v = u+C*z;
        Eigen::VectorXd dv = Puopt_Plamjs + PC_Plamjs*z;
        double dvTSigmav = 2*dv.transpose()*Sigma_inv*v;

        double out = -.5 * dvTSigmav;
        for(int r = 0; r < p; r++){

            // update item object
            Eigen::VectorXd lamr = Lam.row(r);
            item.update_eta_(lambda0(r), lamr, v);
            item.update_out_opt_();

            // helper quantities
            Eigen::VectorXd dlamr(q); dlamr.fill(0.0);
            if(j==r){
                dlamr(s) = 1.0;
            }

            double deta = 0; deta += dlamr.transpose()*v; deta += lamr.transpose()*dv;
            double dtheta = item.br1 * deta;

            out += item.phi*(y(r)-item.b1)*dtheta;
        }

        return out;
    }

    /*Outer optimization wrt lamj0*/
    Eigen::VectorXd Puopt_Plamj0;
    Eigen::VectorXd Puopt_Plamj0_(Eigen::MatrixXd Hinv, int j){
        Puopt_Plamj0.resize(q);

        Eigen::VectorXd lamj = Lam.row(j);
        item.update_eta_(lambda0(j), lamj, u);
        item.update_out_opt_();
        Puopt_Plamj0 = Hinv * (item.phi*(-item.b2*item.Ptheta_Plambdaj0_()*item.Ptheta_Pu_()+(y(j)-item.b1)*item.Ptheta_PuPlambdaj0_()));

        return Puopt_Plamj0;
    }

    Eigen::MatrixXd PH_Plamj0_(Eigen::MatrixXd Hinv, int j){

        Eigen::MatrixXd jac(q,q); jac.fill(0.0);
        for(int r = 0; r < p; r++){

            // contribution of item r
            Eigen::MatrixXd Pr(q,q); Pr.fill(0.0);

            // update item object
            Eigen::VectorXd lamr = Lam.row(r);
            item.update_eta_(lambda0(r), lamr, u);
            item.update_out_opt_();

            // helper quantities
            double dlamr0 = 0;
            if(r == j) dlamr0 = 1;

            double deta = dlamr0; deta += lamr.transpose()*Puopt_Plamj0;
            double dtheta = item.br1 * deta;
            Eigen::MatrixXd dthetauthetauT = 2*item.br1*item.br2*deta*lamr*lamr.transpose();
            Eigen::MatrixXd dthetauuT = item.br3 * deta * lamr*lamr.transpose();

            Pr += -item.b3*dtheta*item.Ptheta_Pu_()*item.Ptheta_Pu_().transpose();
            Pr += -item.b2*dthetauthetauT;
            Pr += -item.b2*dtheta*item.Ptheta_PuPuT_();
            Pr += (y(r)-item.b1)*dthetauuT;

            jac -= item.phi * Pr;
        }
        return jac;

    }

    Eigen::MatrixXd PC_Plamj0;
    Eigen::MatrixXd PC_Plamj0_(Eigen::SparseMatrix<double> C, Eigen::MatrixXd Hinv, int j){
        PC_Plamj0.resize(q,q); PC_Plamj0.fill(0.0);
        Eigen::MatrixXd B(q,q); B.fill(0.0);
        for(int k = 0; k<q; k++){
            for(int l = 0; l<q; l++){
                if(k>l)B(k,l)  = 1;
                if(k<l)B(k,l)  = 0;
                if(k==l)B(k,l) =.5;
            }
        }

        Eigen::MatrixXd tmp1 = C.transpose()*PH_Plamj0_(Hinv, j)*C;
        Eigen::MatrixXd tmp2 = B.array()*tmp1.array();
        PC_Plamj0 = -C*tmp2;
        return PC_Plamj0;
    }

    double PlogdetC_Plamj0_(Eigen::SparseMatrix<double> C, Eigen::MatrixXd Hinv, int j){
        Eigen::MatrixXd Iq; Iq.setIdentity(q, q);
        Eigen::MatrixXd Cinv = C.triangularView<Eigen::Lower>().solve(Iq);

        Eigen::MatrixXd tmp1 = Cinv*PC_Plamj0;
        double g = tmp1.trace();
        return g;
    }

    double Pjllv_Plamj0_(Eigen::SparseMatrix<double> C, Eigen::MatrixXd Hinv, int j, Eigen::VectorXd z){
        Eigen::VectorXd v = u+C*z;
        Eigen::VectorXd dv = Puopt_Plamj0 + PC_Plamj0*z;
        double dvTSigmav = 2*dv.transpose()*Sigma_inv*v;

        double out = -.5 * dvTSigmav;
        for(int r = 0; r < p; r++){

            // update item object
            Eigen::VectorXd lamr = Lam.row(r);
            item.update_eta_(lambda0(r), lamr, v);
            item.update_out_opt_();

            // helper quantities
            double dlamr0 = 0;
            if(r == j) dlamr0 = 1;

            double deta = dlamr0; deta += lamr.transpose()*dv;
            double dtheta = item.br1 * deta;

            out += item.phi*(y(r)-item.b1)*dtheta;
        }

        return out;
    }

    /* Outer optimization wrt rhors */
    Eigen::MatrixXd dS;
    Eigen::MatrixXd PSigma_Ptrhors_(int r, int s, double trhors){
        Eigen::MatrixXd Srs(q,q); Srs.fill(0.0);
        Srs(r,s) = 1; Srs(s,r) = 1;
        double drho = 2*exp(2*trhors) * pow((exp(2*trhors) + 1),-1) * ( 1 - ( exp(2*trhors) - 1) * pow((exp(2*trhors) + 1),-1) );
        dS = drho * Srs;
        return dS;
    }

    Eigen::VectorXd Puopt_Ptrhors;
    Eigen::VectorXd Puopt_Ptrhors_(Eigen::MatrixXd Hinv, int r, int s, double trhors){
        Puopt_Ptrhors.resize(q);

        // Derivative of Sigma
        Puopt_Ptrhors = Hinv * (Sigma_inv*dS*Sigma_inv*u);

        return Puopt_Ptrhors;
    }

    Eigen::MatrixXd PH_Ptrhors_(Eigen::MatrixXd Hinv, int r, int s, double trhors){
        Eigen::MatrixXd dSinv = -Sigma_inv * dS *Sigma_inv;

        Eigen::MatrixXd jac(q,q);
        jac = dSinv;

        for(int j = 0; j < p; j++){

            // contribution of item j
            Eigen::MatrixXd Pj(q,q); Pj.fill(0.0);

            // update item object
            Eigen::VectorXd lamj = Lam.row(j);
            item.update_eta_(lambda0(j), lamj, u);
            item.update_out_opt_();

            // helper quantitities
            double deta = lamj.transpose()*Puopt_Ptrhors;
            double dtheta = item.br1 * deta;
            Eigen::MatrixXd dthetauthetauT = 2*item.br1*item.br2*deta*lamj*lamj.transpose();
            Eigen::MatrixXd dthetauuT = item.br3 * deta * lamj*lamj.transpose();

            Pj += -item.b3*dtheta*item.Ptheta_Pu_()*item.Ptheta_Pu_().transpose();
            Pj += -item.b2*dthetauthetauT;
            Pj += -item.b2*dtheta*item.Ptheta_PuPuT_();
            Pj += (y(j)-item.b1)*dthetauuT;

            jac -= item.phi * Pj;
        }
        return jac;

    }

    Eigen::MatrixXd PC_Ptrhors;
    Eigen::MatrixXd PC_Ptrhors_(Eigen::SparseMatrix<double> C, Eigen::MatrixXd Hinv, int r, int s, double trhors){
        PC_Ptrhors.resize(q,q); PC_Ptrhors.fill(0.0);
        Eigen::MatrixXd B(q,q); B.fill(0.0);
        for(int k = 0; k<q; k++){
            for(int l = 0; l<q; l++){
                if(k>l)B(k,l)  = 1;
                if(k<l)B(k,l)  = 0;
                if(k==l)B(k,l) =.5;
            }
        }

        Eigen::MatrixXd tmp1 = C.transpose()*PH_Ptrhors_(Hinv, r, s, trhors)*C;
        Eigen::MatrixXd tmp2 = B.array()*tmp1.array();
        PC_Ptrhors = -C*tmp2;
        return PC_Ptrhors;
    }

    double PlogdetC_Ptrhors_(Eigen::SparseMatrix<double> C, Eigen::MatrixXd Hinv, int r, int s, double trhors){
        Eigen::MatrixXd Iq; Iq.setIdentity(q, q);
        Eigen::MatrixXd Cinv = C.triangularView<Eigen::Lower>().solve(Iq);

        Eigen::MatrixXd tmp1 = Cinv*PC_Ptrhors;
        double gr = tmp1.trace();
        return gr;
    }

    double Pjllv_Ptrhors_(Eigen::SparseMatrix<double> C, Eigen::MatrixXd Hinv, int r, int s, double trhors, Eigen::VectorXd z){
        Eigen::VectorXd v = u+C*z;
        Eigen::VectorXd dv = Puopt_Ptrhors + PC_Ptrhors*z;
        Eigen::MatrixXd dSigma_inv = -Sigma_inv * dS * Sigma_inv;
        double dvTSigmav = 0; dvTSigmav += dv.transpose()*Sigma_inv*v; dvTSigmav += v.transpose()*(dSigma_inv*v+Sigma_inv*dv);
        double dlogdetSigma = (Sigma_inv*dS).trace();

        double out = -.5 * (dlogdetSigma + dvTSigmav);
        for(int j = 0; j < p; j++){

            // update item object
            Eigen::VectorXd lamj = Lam.row(j);
            item.update_eta_(lambda0(j), lamj, v);
            item.update_out_opt_();

            double deta = lamj.transpose()*dv;
            double dtheta = item.br1 * deta;

            out += item.phi*(y(j)-item.b1)*dtheta;
        }

        return out;
    }

};

// class for Inner Newton optimization
class NewtonOpt{
public:

    // input class
    joint_nll jnlli;
    int maxiter;
    double tol;

    // intermediate quantities
    int iter;
    double last_tol;

    void setup_(joint_nll jnll_input, int maxiter_input, double tol_input){
        jnlli = jnll_input;
        maxiter = maxiter_input;
        tol = tol_input;
    }

    void optimize_(Eigen::VectorXd &u, Eigen::MatrixXd &H){
        u.fill(0);
        int q = u.size();
        iter = 0;
        last_tol = 1;

        Eigen::MatrixXd Iq; Iq.setIdentity(q, q);
        double jnll_previous = 0;
        jnlli.update_u_(u);

        while(iter < maxiter && last_tol > tol){
            jnll_previous = jnlli.jnll_();
            jnlli.der_u_();
            Eigen::VectorXd g = jnlli.gr_u;
            Eigen::MatrixXd h = jnlli.h_u;
            Eigen::LLT<Eigen::MatrixXd> llt(h);
            Eigen::MatrixXd Q = llt.solve(Iq);
            u = u - Q*g;

            jnlli.update_u_(u);
            last_tol = std::abs(jnlli.jnll_() - jnll_previous);
            iter ++;
        }

        jnlli.update_u_(u);
        jnlli.der_u_();
        H = jnlli.h_u; //not -H, because it's coded wrt to the joint negative ll
        last_tol = std::abs(jnlli.jnll_() - jnll_previous);
    }
};

#endif
