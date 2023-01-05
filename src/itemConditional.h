#ifndef itemConditional_H
#define itemConditional_H

// class specific for binary items
class binary_item{
public:
    // class inputs
    double lambda0;
    Eigen::VectorXd lambda;
    Eigen::VectorXd u;
    double phi = 1;
    int linkFLAG; // 0 canonical link (logit), 1 probit

    // class outputs
    double c = 0;
    double mu;      double mu1;    double mu2;    double mu3;
    double theta;   double theta1; double theta2; double theta3;
    double b = 0;   double b1;     double b2;     double b3;
    double eta ;
    double br1;     double br2;    double br3;

    // update function:
    // 1. store new eta; 2. update mu (link dependently).
    void update_eta_(double lambda0_input,
                     Eigen::VectorXd lambda_input,
                     Eigen::VectorXd u_input){

        lambda0 = lambda0_input;
        lambda = lambda_input;
        u = u_input;

        // update eta
        eta = 0;
        eta = lambda0 + lambda.transpose()*u;

        // update mean response
        if(linkFLAG == 0){
            mu = exp(eta)/(1 + exp(eta));
        } else if(linkFLAG == 1){
            mu = R::pnorm(eta, 0.0, 1.0, 1, 0);
        }

    }

    // exponential family quantities for binary item
    double theta_(){
        theta = 0;
        theta = log(mu/(1-mu));
        return theta;}
    double b_(){
        b = 0;
        b = log(1+exp(theta));
        return b;}
    double c_(){
        return c;}

    // computing nll
    void update_nll_(){
        theta_();
        b_();
        c_();
    }
    double nll_(double y){
        double nll;
        nll = -((y*theta-b)/phi) - c;
        return nll;
    }

    // derivatives of b wrt theta
    double b1_(){
        b1 = exp(theta)/(1+exp(theta));
        return b1;
    }
    double b2_(){
        b2 = exp(theta)/(pow(1+exp(theta),2));
        return b2;
    }
    double b3_(){
        b3 = exp(theta)*(1-exp(theta))/(pow(1+exp(theta),3));
        return b3;
    }

    // derivatives of theta wrt mu
    double theta1_(){
        theta1 = 1/(mu*(1-mu));
        return theta1;
    }
    double theta2_(){
        theta2 = (2*mu - 1)/(pow(mu,2)*pow(1-mu,2));
        return theta2;
    }
    double theta3_(){
        theta3 = 2*pow((mu*(1-mu)), -2)*(1+pow(2*mu-1,2)*pow((mu*(1-mu)), -1));
        return theta3;
    }

    // derivatives of mu wrt eta (link dependent)
    double mu1_(){
        if(linkFLAG == 0){
            mu1 = 1/theta1;
        } else if(linkFLAG == 1){
            mu1 = R::dnorm(eta, 0, 1, 0);
        }

        return mu1;
    }
    double mu2_(){
        if(linkFLAG == 0){
            mu2 = mu1 - 2*exp(2*eta)*pow(1+exp(eta),-2)+2*exp(3*eta)*pow(1+exp(eta),-3);
        } else if(linkFLAG == 1){
            mu2 = -mu1*eta;
        }

        return mu2;
    }
    double mu3_(){
        if(linkFLAG == 0){
            mu3 = mu2 - 4*exp(2*eta)*pow(1+exp(eta),-2)+10*exp(3*eta)*pow(1+exp(eta),-3)-6*exp(4*eta)*pow(1+exp(eta),-4);
        } else if(linkFLAG == 1){
            mu3 = -mu2*eta - mu1;
        }

        return mu3;
    }

};

// generic class for exponential family items
// Takes binary items as input
class ef_item{
public:

    // class inputs
    double lambda0;
    double phi = 1;
    Eigen::VectorXd lambda;
    Eigen::VectorXd u;
    binary_item input_item;
    int linkFLAG;

    // inheritance
    double c;
    double mu;      double mu1;    double mu2;    double mu3;
    double theta;   double theta1; double theta2; double theta3;
    double b;       double b1;     double b2;     double b3;
    double eta ;
    double br1;     double br2;    double br3;

    // update function:
    // 1. store new eta; 2. update mu (link dependently).
    void update_eta_(double lambda0_input,
                     Eigen::VectorXd lambda_input,
                     Eigen::VectorXd u_input){

        lambda0 = lambda0_input;
        lambda = lambda_input;
        u = u_input;

        input_item.linkFLAG = linkFLAG;
        input_item.update_eta_(lambda0, lambda, u);
        eta = input_item.eta;
        mu = input_item.mu;

    }

    // computing nll
    void update_nll_(){
        //input_item.update_nll_();
        theta = input_item.theta_();
        b = input_item.b_();
        c = input_item.c_();
    }
    double nll_(double y){
        double nll;
        nll = -((y*theta-b)/phi) - c;
        return nll;
    }

    /* Inner Optimization operations */
    void update_in_opt_(){
        b1 = input_item.b1_();
        b2 = input_item.b2_();
        theta1 = input_item.theta1_();
        theta2 = input_item.theta2_();
        mu1 = input_item.mu1_();
        mu2 = input_item.mu2_();

        // exporting scalar quantities depending on eta used in derivatives
        br1 = theta1 * mu1;
        br2 = theta2*pow(mu1,2) + theta1*mu2;
    }

    // chain rule derivatives of theta wrt u
    Eigen::VectorXd Ptheta_Pu_(){
        Eigen::VectorXd g(lambda.size());
        g = br1 * lambda;
        return g;
    }

    Eigen::MatrixXd Ptheta_PuPuT_(){
        Eigen::MatrixXd h(lambda.size(), lambda.size());
        h = br2*lambda*lambda.transpose();
        return h;
    }

    // chain derivatives wrt lambdajs
    double Ptheta_Plambdajs_(int s){
        double g = br1 * u(s);
        return g;
    }
    Eigen::VectorXd Ptheta_PuPlambdajs_(int s){
        int q = lambda.size();
        Eigen::VectorXd g(q);
        Eigen::VectorXd es(q); es.fill(0.0); es(s) = 1;
        g = br2*u(s)*lambda + br1*es;

        return g;

    }

    // chain derivatives wrt intercept j
    double Ptheta_Plambdaj0_(){
        double g = br1;
        return g;
    }
    Eigen::VectorXd Ptheta_PuPlambdaj0_(){
        int q = lambda.size();
        Eigen::VectorXd g(q);
        g = br2*lambda;
        return g;
    }

    void update_out_opt_(){
        update_nll_();
        update_in_opt_();

        // 3rd order derivatives
        mu3 = input_item.mu3_();
        theta3 = input_item.theta3_();
        b3 = input_item.b3_();
        br3 = theta3*pow(mu1,3) + 3*theta2*mu1*mu2 +theta1*mu3;
    }


};


#endif
