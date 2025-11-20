#include "gprs/sparse_gpr.h"
#include <iostream>
#include <Eigen/Dense>

namespace GPRcpp
{

void SparseGPR::fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train, const Eigen::MatrixXd & inducing_points)
{
    std::cout << "[GPR INFO]: fitting sparse GPR with method " << inference_method << '\n';
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
    std::cout << "<!!!! It is sparse GPR, please use fit function with inducing points !!!!>" << '\n';
    throw std::runtime_error("Not implemented");
}

void SparseGPR::fit_with_dtc()
{
    double inv_sigma = 1.0 / fmax(likelihood_varience, alpha_);

    // Get Kuu and Luu from the inducing points
    Eigen::MatrixXd inv_sigma_Y = inv_sigma * y_train_; // (Y - mean) if mean is not 0
    // double trYYT = inv_sigma_Y.squaredNorm(); // What this for?
    Eigen::MatrixXd Kuu = kernel_->evaluate(m_inducing_point);
    Kuu = (Kuu + Kuu.transpose()) / 2 + Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols()) * alpha_;

    Eigen::LLT<Eigen::MatrixXd> LowerK_uu = Kuu.llt();

    // Compute psi stats and factor B
    Eigen::MatrixXd Kfu = kernel_->evaluate(X_train_, m_inducing_point);
    Eigen::MatrixXd tmp1 = Kfu * sqrt(inv_sigma);
    Eigen::MatrixXd tmp2 = LowerK_uu.matrixL().solve(tmp1.transpose());
    Eigen::MatrixXd B = tmp2 * tmp2.transpose() + Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols()); // If you want to compute log margin, seperate this
    
    Eigen::LLT<Eigen::MatrixXd> LB_LLT;
    LB_LLT = B.llt();
    Eigen::MatrixXd Bi = -1.0 * LB_LLT.solve(Eigen::MatrixXd::Identity(B.rows(), B.cols())) + Eigen::MatrixXd::Identity(B.rows(), B.cols());

    // compute woodbury inv and vector
    Eigen::MatrixXd tmp3 = LowerK_uu.matrixL().solve(Kfu.transpose());
    Eigen::MatrixXd LBi_Lmi_K_fu = LB_LLT.matrixL().solve(tmp3);
    Eigen::MatrixXd LBi_Lmi_K_fu_vf = LBi_Lmi_K_fu * inv_sigma_Y;
    Eigen::MatrixXd tmp4 = LB_LLT.matrixU().solve(LBi_Lmi_K_fu_vf);
    Alpha_ = LowerK_uu.matrixU().solve(tmp4);
    woodbury_inv = (LowerK_uu.matrixU().solve((LowerK_uu.matrixU().solve(Bi)).transpose())).transpose();


    // For debug
    // std::cout << "\nInducing point is:\n" << m_inducing_point << '\n';
    // std::cout << "\nneg_sigma_Y is:\n" << inv_sigma_Y << '\n';
    // std::cout << "\ntrYYT is:\n" << trYYT << '\n';
    // std::cout << "\nK_uu is:\n" << Kuu << '\n';
    // std::cout << "\nLm is:\n" << LowerK_uu.matrixL().toDenseMatrix().block(0, 0, 4, 4) << '\n';
    // std::cout << "\nLB is:\n" << LB_LLT.matrixL().toDenseMatrix().block(0, 0, 4, 4) << '\n';
    // std::cout << "\nK_fu is:\n" << Kfu << '\n';
    // std::cout << "\ntmp1 is:\n" << tmp1 << '\n';
    // std::cout << "\ntmp2 is:\n" << tmp2 << '\n';
    // std::cout << "\nB is:\n" << B << '\n';
    // std::cout << "\nLB is:\n" << LB << '\n';
    // std::cout << "\ntmp3 is:\n" << tmp3 << '\n';
    // std::cout << "\nLBi_Lmi_K_fu is:\n" << LBi_Lmi_K_fu << '\n';
    // std::cout << "\nLBi_Lmi_K_fu_vf is:\n" << LBi_Lmi_K_fu_vf << '\n';
    // std::cout << "\ntmp4 is:\n" << tmp4 << '\n';
    // std::cout << "\nAlpha_ is:\n" << Alpha_ << '\n';
    // std::cout << "\nBi is:\n" << Bi << '\n';
    // std::cout << "\nwoodbury_inv is:\n" << woodbury_inv << '\n';
}

void SparseGPR::fit_with_fitc()
{
    /*
     * Compare with the pyGPR code, Luu_inv_Kuf => W = Luu_inv @ Kuf,
     * beta_star => diag_inv, LA => L, URiy => Kuf @ diag_inv @ y,
     * Alpha_ => Luu.T.inv @ L.T.inv @ L.inv @ Luu.inv @ URiy (L.inv @ Luu.inv @ URiy = L_inv_W_d_inv_y)
     * */
    alpha_ = 1e-6;
    double sigma_n = likelihood_varience;
    Eigen::MatrixXd Kuu = kernel_->evaluate(m_inducing_point);
    Eigen::MatrixXd Kff = kernel_->k_diag(X_train_); // FIRST IMPLEMENT THIS
    Eigen::MatrixXd Kfu = kernel_->evaluate(X_train_, m_inducing_point);
    Kuu += Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols()) * alpha_;
    Eigen::MatrixXd Luu, Kuu_inv;

    Eigen::LLT<Eigen::MatrixXd> LowerK_uu = Kuu.llt();
    Luu = LowerK_uu.matrixL();
    Kuu_inv = LowerK_uu.solve(Eigen::MatrixXd::Identity(Luu.rows(), Luu.cols()));

    Eigen::MatrixXd Luu_inv = Luu.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(Luu.rows(), Luu.cols()));
    
    // compute beta_star
    Eigen::MatrixXd Luu_inv_Kuf = Luu_inv * Kfu.transpose();
    Eigen::MatrixXd beta_star = 1.0 / (Kff.array() + sigma_n - Luu_inv_Kuf.colwise().squaredNorm().transpose().array());// - Luu_inv_Kuf.colwise().squaredNorm().array();Kff.array() + sigma_n
    Eigen::MatrixXd beta_star_sqrt = Luu_inv_Kuf.array() * beta_star.array().sqrt().transpose().replicate(Luu_inv_Kuf.rows(), 1);
    Eigen::MatrixXd A = beta_star_sqrt * beta_star_sqrt.transpose() + Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols());
    Eigen::MatrixXd LA;

    Eigen::LLT<Eigen::MatrixXd> LA_LLT = A.llt();
    LA = LA_LLT.matrixL();

    // back substutue
    Eigen::MatrixXd URiy = (Kfu.array() * beta_star.array().replicate(1, Kfu.cols())).matrix().transpose() * y_train_; //beta_star.array().replicate() Kfu.array().transpose()
    Eigen::MatrixXd tmp1 = Luu.triangularView<Eigen::Lower>().solve(URiy);
    Eigen::MatrixXd B = LA.triangularView<Eigen::Lower>().solve(tmp1);
    Eigen::MatrixXd tmp2 = LA.transpose().triangularView<Eigen::Upper>().solve(B);
    Alpha_ = Luu.transpose().triangularView<Eigen::Upper>().solve(tmp2);
    Eigen::MatrixXd tmp3 = LA.triangularView<Eigen::Lower>().solve(Luu_inv);
    Eigen::MatrixXd P = tmp3.transpose() * tmp3;
    woodbury_inv = Kuu_inv - P;

    // std::cout << "\nK_uu is:\n" << Kuu << '\n';
    // std::cout << "\nKnn is:\n" << Kff << '\n';
    // std::cout << "\nKnm is:\n" << Kfu << '\n';
    // std::cout << "\nLm is:\n" << Luu << '\n';
    // std::cout << "\nLmi is:\n" << Luu_inv << '\n';
    // std::cout << "\nK_uui is:\n" << Kuu_inv << '\n';
    // std::cout << "\nLiUT is:\n" << Luu_inv_Kuf << '\n';
    // std::cout << "\nbeta_star is:\n" << beta_star << '\n';
    // std::cout << "\nbeta_star_sqrt is:\n" << beta_star_sqrt << '\n';
    // std::cout << "\nA is:\n" << A << '\n';
    // std::cout << "\nLA is:\n" << LA << '\n';
    // std::cout << "\nURiy is:\n" << URiy << '\n';
    // std::cout << "\ntmp1 is:\n" << tmp1 << '\n';
    // std::cout << "\nB is:\n" << B << '\n';
    // std::cout << "\ntmp2 is:\n" << tmp2 << '\n';
    // std::cout << "\nV is:\n" << Alpha_ << '\n';
    // std::cout << "\ntmp3 is:\n" << tmp3 << '\n';
    // std::cout << "\nP is:\n" << P << '\n';
    // std::cout << "\nwoodbury_inv is:\n" << woodbury_inv << '\n';
}

gpr_results SparseGPR::predict(const Eigen::MatrixXd & X_test)
{
    return predict(X_test, false);
}

gpr_results SparseGPR::predict(const Eigen::MatrixXd & X_test, bool return_cov, bool compute_jac)
{
    if (!has_x_train_)
    {
        results_.y_mean = Eigen::VectorXd::Zero(X_test.rows());
        results_.y_cov = Eigen::MatrixXd::Constant(X_test.rows(), X_test.rows(), 1.0);
        std::cout << "No training data:\n" << kernel_->evaluate(X_test) << '\n';
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

        if (compute_jac)
        {
            const Eigen::MatrixXd dk_dx =  kernel_->dk_dx(m_inducing_point, X_test);
            results_.dmu_dx = dk_dx.transpose() * Alpha_;
            if (normalize_y_)
            {
                results_.dmu_dx = (results_.dmu_dx.array().rowwise() * y_train_std_.array());
            }
        }

        if (normalize_y_)
        {
            results_.y_mean = (results_.y_mean.array().rowwise() * y_train_std_.array()).rowwise() + y_train_mean_.array();
            results_.y_cov = results_.y_cov * y_train_std_(0) * y_train_std_(0); // Only for 1D output.
            // In sklearn, for 2D output, the K_trans is the same, the y_cov is just [y_cov * std_1, y_cov * std_2]
        }

        // std::cout << "K_trans:\n" << K_trans << '\n';
        // std::cout << "y_mean:\n" << results_.y_mean << '\n';
        // std::cout << "V:\n" << V << '\n';
        // std::cout << "cov:\n" << results_.y_cov << '\n';
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
    if (normalize_y_)
    {
        certain_predict.dmu_dx = (certain_predict.dmu_dx.array().rowwise() * y_train_std_.array());
    }

    double first_order_varience = (certain_predict.dmu_dx.transpose() * input_cov * certain_predict.dmu_dx)(0);

    if (add_covariance)
    {
        certain_predict.y_covariance = certain_predict.dmu_dx.transpose() * input_cov;
    }

    if (normalize_y_)
    {
        first_order_varience = first_order_varience * y_train_std_(0) * y_train_std_(0); // Only for 1D output.
        if (add_covariance) certain_predict.y_covariance = certain_predict.y_covariance * y_train_std_(0) * y_train_std_(0);
    }

    certain_predict.y_cov(0, 0) += first_order_varience;

    // std::cout << "[ExactGPR]: dmu_dx.T * input_cov is:\n" << certain_predict.y_covariance << '\n';
    // std::cout << "[ExactGPR]: dk_dx is:\n" << dk_dx << '\n';
    // std::cout << "[ExactGPR]: Alpha_ is:\n" << Alpha_ << '\n';
    // std::cout << "[ExactGPR]: dmu_dx is:\n" << dmu_dx << '\n';
    // std::cout << "[ExactGPR]: first_order_varience is:\n" << first_order_varience << '\n';
    // std::cout << "[ExactGPR]: exact cov is:\n" << certain_predict.y_cov << '\n';

    return certain_predict;
}

SparseGPR::~SparseGPR()
{

}

} // namespace GPRcpp
