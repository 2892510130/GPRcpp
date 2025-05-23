#include "gprs/sparse_gpr.h"
#include <iostream>
#include <Eigen/Dense>

namespace GPRcpp
{

void SparseGPR::fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train, const Eigen::MatrixXd & inducing_points)
{
    std::cout << "<***** Fitting Sparse GPR *****>" << std::endl;
    has_x_train_ = true;
    X_train_ = X_train;
    y_train_ = y_train;
    m_inducing_point = inducing_points;

    if (normalize_y_)
    {
        y_train_mean_ = y_train_.colwise().mean();
        y_train_std_ = ((y_train_.rowwise() - y_train_mean_).array().square().colwise().sum() / y_train_.rows()).sqrt();
        y_train_ = (y_train_.rowwise() - y_train_mean_).array().rowwise() / y_train_std_.array();
    }

    if (inference_method == 0) fit_with_dtc();
    else if (inference_method == 1) fit_with_fitc();
}

void SparseGPR::fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train)
{
    std::cout << "<!!!! It is sparse GPR, please use fit function with inducing points !!!!>" << std::endl;
    throw std::runtime_error("Not implemented");
}

void SparseGPR::fit_with_dtc()
{
    double precision = 1.0 / fmax(likelihood_varience, alpha_);

    // Get Kmm and Lm from the inducing points
    Eigen::MatrixXd VVT_factor = precision * y_train_; // (Y - mean) if mean is not 0
    double trYYT = VVT_factor.squaredNorm();
    Eigen::MatrixXd Kmm = kernel_->evaluate(m_inducing_point);
    Kmm += Eigen::MatrixXd::Identity(Kmm.rows(), Kmm.cols()) * alpha_;


    Eigen::LLT<Eigen::MatrixXd> Lm_LLT = Kmm.llt();

    // Compute psi stats and factor B
    Eigen::MatrixXd psi1 = kernel_->evaluate(X_train_, m_inducing_point);
    Eigen::MatrixXd tmp1 = psi1 * sqrt(precision);
    Eigen::MatrixXd tmp2 = Lm_LLT.matrixL().solve(tmp1.transpose());
    Eigen::MatrixXd B = tmp2 * tmp2.transpose() + Eigen::MatrixXd::Identity(Kmm.rows(), Kmm.cols()); // If you want to compute log margin, seperate this
    
    Eigen::LLT<Eigen::MatrixXd> LB_LLT;
    LB_LLT = B.llt();
    Eigen::MatrixXd Bi = -1.0 * LB_LLT.solve(Eigen::MatrixXd::Identity(B.rows(), B.cols())) + Eigen::MatrixXd::Identity(B.rows(), B.cols());

    // compute woodbury inv and vector
    Eigen::MatrixXd tmp3 = Lm_LLT.matrixL().solve(psi1.transpose());
    Eigen::MatrixXd LBi_Lmi_psi1 = LB_LLT.matrixL().solve(tmp3);
    Eigen::MatrixXd LBi_Lmi_psi1_vf = LBi_Lmi_psi1 * VVT_factor;
    Eigen::MatrixXd tmp4 = LB_LLT.matrixU().solve(LBi_Lmi_psi1_vf);
    Alpha_ = Lm_LLT.matrixU().solve(tmp4);
    woodbury_inv = (Lm_LLT.matrixU().solve((Lm_LLT.matrixU().solve(Bi)).transpose())).transpose();


    // For debug
    // std::cout << "\nInducing point is:\n" << m_inducing_point << std::endl;
    // std::cout << "\nVVT_factor is:\n" << VVT_factor << std::endl;
    // std::cout << "\ntrYYT is:\n" << trYYT << std::endl;
    // std::cout << "\nKmm is:\n" << Kmm << std::endl;
    // std::cout << "\nLm is:\n" << Lm_LLT.matrixL().toDenseMatrix().block(0, 0, 4, 4) << std::endl;
    // std::cout << "\nLB is:\n" << LB_LLT.matrixL().toDenseMatrix().block(0, 0, 4, 4) << std::endl;
    // std::cout << "\npsi1 is:\n" << psi1 << std::endl;
    // std::cout << "\ntmp1 is:\n" << tmp1 << std::endl;
    // std::cout << "\ntmp2 is:\n" << tmp2 << std::endl;
    // std::cout << "\nB is:\n" << B << std::endl;
    // std::cout << "\nLB is:\n" << LB << std::endl;
    // std::cout << "\ntmp3 is:\n" << tmp3 << std::endl;
    // std::cout << "\nLBi_Lmi_psi1 is:\n" << LBi_Lmi_psi1 << std::endl;
    // std::cout << "\nLBi_Lmi_psi1_vf is:\n" << LBi_Lmi_psi1_vf << std::endl;
    // std::cout << "\ntmp4 is:\n" << tmp4 << std::endl;
    // std::cout << "\nAlpha_ is:\n" << Alpha_ << std::endl;
    // std::cout << "\nBi is:\n" << Bi << std::endl;
    // std::cout << "\nwoodbury_inv is:\n" << woodbury_inv << std::endl;
}

void SparseGPR::fit_with_fitc()
{
    alpha_ = 1e-6;
    double sigma_n = likelihood_varience;
    Eigen::MatrixXd Kmm = kernel_->evaluate(m_inducing_point);
    Eigen::MatrixXd Knn = kernel_->k_diag(X_train_); // FIRST IMPLEMENT THIS
    Eigen::MatrixXd Knm = kernel_->evaluate(X_train_, m_inducing_point);
    Kmm += Eigen::MatrixXd::Identity(Kmm.rows(), Kmm.cols()) * alpha_;
    Eigen::MatrixXd Lm, Kmmi;

    Eigen::LLT<Eigen::MatrixXd> Lm_LLT = Kmm.llt();
    Lm = Lm_LLT.matrixL();
    Kmmi = Lm_LLT.solve(Eigen::MatrixXd::Identity(Lm.rows(), Lm.cols()));

    Eigen::MatrixXd Lmi = Lm.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(Lm.rows(), Lm.cols()));
    
    // compute beta_star
    Eigen::MatrixXd LiUT = Lmi * Knm.transpose();
    Eigen::MatrixXd beta_star = 1.0 / (Knn.array() + sigma_n - LiUT.colwise().squaredNorm().transpose().array());// - LiUT.colwise().squaredNorm().array();Knn.array() + sigma_n
    Eigen::MatrixXd beta_star_sqrt = LiUT.array() * beta_star.array().sqrt().transpose().replicate(LiUT.rows(), 1);
    Eigen::MatrixXd A = beta_star_sqrt * beta_star_sqrt.transpose() + Eigen::MatrixXd::Identity(Kmm.rows(), Kmm.cols());
    Eigen::MatrixXd LA;

    Eigen::LLT<Eigen::MatrixXd> LA_LLT = A.llt();
    LA = LA_LLT.matrixL();

    // back substutue
    Eigen::MatrixXd URiy = (Knm.array() * beta_star.array().replicate(1, Knm.cols())).matrix().transpose() * y_train_; //beta_star.array().replicate() Knm.array().transpose()
    Eigen::MatrixXd tmp1 = Lm.triangularView<Eigen::Lower>().solve(URiy);
    Eigen::MatrixXd B = LA.triangularView<Eigen::Lower>().solve(tmp1);
    Eigen::MatrixXd tmp2 = LA.transpose().triangularView<Eigen::Upper>().solve(B);
    Alpha_ = Lm.transpose().triangularView<Eigen::Upper>().solve(tmp2);
    Eigen::MatrixXd tmp3 = LA.triangularView<Eigen::Lower>().solve(Lmi);
    Eigen::MatrixXd P = tmp3.transpose() * tmp3;
    woodbury_inv = Kmmi - P;

    // std::cout << "\nKmm is:\n" << Kmm << std::endl;
    // std::cout << "\nKnn is:\n" << Knn << std::endl;
    // std::cout << "\nKnm is:\n" << Knm << std::endl;
    // std::cout << "\nLm is:\n" << Lm << std::endl;
    // std::cout << "\nLmi is:\n" << Lmi << std::endl;
    // std::cout << "\nKmmi is:\n" << Kmmi << std::endl;
    // std::cout << "\nLiUT is:\n" << LiUT << std::endl;
    // std::cout << "\nbeta_star is:\n" << beta_star << std::endl;
    // std::cout << "\nbeta_star_sqrt is:\n" << beta_star_sqrt << std::endl;
    // std::cout << "\nA is:\n" << A << std::endl;
    // std::cout << "\nLA is:\n" << LA << std::endl;
    // std::cout << "\nURiy is:\n" << URiy << std::endl;
    // std::cout << "\ntmp1 is:\n" << tmp1 << std::endl;
    // std::cout << "\nB is:\n" << B << std::endl;
    // std::cout << "\ntmp2 is:\n" << tmp2 << std::endl;
    // std::cout << "\nV is:\n" << Alpha_ << std::endl;
    // std::cout << "\ntmp3 is:\n" << tmp3 << std::endl;
    // std::cout << "\nP is:\n" << P << std::endl;
    // std::cout << "\nwoodbury_inv is:\n" << woodbury_inv << std::endl;
}

gpr_results SparseGPR::predict(const Eigen::MatrixXd & X_test)
{
    return predict(X_test, false);
}

gpr_results SparseGPR::predict(const Eigen::MatrixXd & X_test, bool return_cov)
{
    if (!has_x_train_)
    {
        results_.y_mean = Eigen::VectorXd::Zero(X_test.rows());
        results_.y_cov = Eigen::MatrixXd::Constant(X_test.rows(), X_test.rows(), 1.0);
        std::cout << "No training data:\n" << kernel_->evaluate(X_test) << std::endl;
        return results_;
    }
    else
    {
        auto K_trans = kernel_->evaluate(X_test, m_inducing_point);
        results_.y_mean = K_trans * Alpha_;

        if (return_cov)
        {
            results_.y_cov = kernel_->evaluate(X_test) - K_trans * woodbury_inv * K_trans.transpose();
        }

        if (normalize_y_)
        {
            results_.y_mean = (results_.y_mean.array().rowwise() * y_train_std_.array()).rowwise() + y_train_mean_.array();
            results_.y_cov = results_.y_cov * y_train_std_(0) * y_train_std_(0); // Only for 1D output.
            // In sklearn, for 2D output, the K_trans is the same, the y_cov is just [y_cov * std_1, y_cov * std_2]
        }

        // std::cout << "K_trans:\n" << K_trans << std::endl;
        // std::cout << "y_mean:\n" << results_.y_mean << std::endl;
        // std::cout << "V:\n" << V << std::endl;
        // std::cout << "cov:\n" << results_.y_cov << std::endl;
        return results_;
    }
}

gpr_results SparseGPR::predict_at_uncertain_input(const Eigen::MatrixXd & X_test, const Eigen::MatrixXd & input_cov)
{
   return predict_at_uncertain_input(X_test, input_cov, false, false);
}

gpr_results SparseGPR::predict_at_uncertain_input(const Eigen::MatrixXd & X_test, const Eigen::MatrixXd & input_cov, bool add_covariance, bool add_second_order_variance)
{
    if (X_test.rows() != 1) throw std::runtime_error("SparseGPR::predict_at_uncertain_input only support 1 sample input!");

    gpr_results certain_predict = predict(X_test, true);

    const Eigen::MatrixXd dk_dx =  kernel_->dk_dx(m_inducing_point, X_test);

    certain_predict.dmu_dx = dk_dx.transpose() * Alpha_;

    double first_order_varience = (certain_predict.dmu_dx.transpose() * input_cov * certain_predict.dmu_dx)(0);

    if (add_covariance)
    {
        certain_predict.y_covariance = certain_predict.dmu_dx.transpose() * input_cov;
    }

    if (normalize_y_)
    {
        first_order_varience = first_order_varience * y_train_std_(0) * y_train_std_(0); // Only for 1D output.
        certain_predict.y_covariance = certain_predict.y_covariance * y_train_std_(0) * y_train_std_(0);
    }

    certain_predict.y_cov(0, 0) += first_order_varience;

    // std::cout << "[ExactGPR]: dmu_dx.T * input_cov is:\n" << certain_predict.y_covariance << std::endl;
    // std::cout << "[ExactGPR]: dk_dx is:\n" << dk_dx << std::endl;
    // std::cout << "[ExactGPR]: Alpha_ is:\n" << Alpha_ << std::endl;
    // std::cout << "[ExactGPR]: dmu_dx is:\n" << dmu_dx << std::endl;
    // std::cout << "[ExactGPR]: first_order_varience is:\n" << first_order_varience << std::endl;
    // std::cout << "[ExactGPR]: exact cov is:\n" << certain_predict.y_cov << std::endl;

    return certain_predict;
}

SparseGPR::~SparseGPR()
{

}

} // namespace GPRcpp