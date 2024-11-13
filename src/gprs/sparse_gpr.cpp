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

    if (sparse_method == 0) fit_with_dtc();
    else if (sparse_method == 1) fit_with_fitc();
}

void SparseGPR::fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train)
{
    std::cout << "<!!!! It is sparse GPR, please use fit function with inducing points !!!!>" << std::endl;
}

void SparseGPR::fit_with_dtc()
{
    double precision = 1.0 / fmax(1.0, alpha_); // TODO: Change it after test (JBL)

    // Get Kmm and Lm from the inducing points
    Eigen::MatrixXd VVT_factor = precision * y_train_; // (Y - mean) if mean is not 0
    double trYYT = VVT_factor.squaredNorm();
    Eigen::MatrixXd Kmm = kernel_->evaluate(m_inducing_point);
    Kmm += Eigen::MatrixXd::Identity(Kmm.rows(), Kmm.cols()) * alpha_;
    Eigen::MatrixXd Lm = Kmm.llt().matrixL();

    // Compute psi stats and factor B
    Eigen::MatrixXd psi1 = kernel_->evaluate(X_train_, m_inducing_point);
    Eigen::MatrixXd tmp1 = psi1 * sqrt(precision);
    Eigen::MatrixXd tmp2 = Lm.triangularView<Eigen::Lower>().solve(tmp1.transpose());
    Eigen::MatrixXd B = tmp2 * tmp2.transpose() + Eigen::MatrixXd::Identity(Kmm.rows(), Kmm.cols()); // If you want to compute log margin, seperate this
    Eigen::MatrixXd LB = B.llt().matrixL();

    // compute woodbury inv and vector
    Eigen::MatrixXd tmp3 = Lm.triangularView<Eigen::Lower>().solve(psi1.transpose());
    Eigen::MatrixXd LBi_Lmi_psi1 = LB.triangularView<Eigen::Lower>().solve(tmp3);
    Eigen::MatrixXd LBi_Lmi_psi1_vf = LBi_Lmi_psi1 * VVT_factor;
    Eigen::MatrixXd tmp4 = LB.transpose().triangularView<Eigen::Upper>().solve(LBi_Lmi_psi1_vf);
    Alpha_ = Lm.transpose().triangularView<Eigen::Upper>().solve(tmp4);
    Eigen::MatrixXd Bi = -1.0 * LB.transpose().triangularView<Eigen::Upper>().solve(
            LB.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(LB.rows(), LB.cols()))) + Eigen::MatrixXd::Identity(LB.rows(), LB.cols());
    woodbury_inv = (Lm.transpose().triangularView<Eigen::Upper>().solve((Lm.transpose().triangularView<Eigen::Upper>().solve(Bi)).transpose())).transpose();
    

    // For debug
    // std::cout << "\nInducing point is:\n" << m_inducing_point << std::endl;
    // std::cout << "\nVVT_factor is:\n" << VVT_factor << std::endl;
    // std::cout << "\ntrYYT is:\n" << trYYT << std::endl;
    // std::cout << "\nKmm is:\n" << Kmm << std::endl;
    // std::cout << "\nLm is:\n" << Lm << std::endl;
    // std::cout << "\npsi1 is:\n" << psi1 << std::endl;
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
    double sigma_n = 1.0; // TODO: make it changable (JBL)
    Eigen::MatrixXd Kmm = kernel_->evaluate(m_inducing_point);
    Eigen::MatrixXd Knn = kernel_->k_diag(X_train_); // FIRST IMPLEMENT THIS
    Eigen::MatrixXd Knm = kernel_->evaluate(X_train_, m_inducing_point);
    Kmm += Eigen::MatrixXd::Identity(Kmm.rows(), Kmm.cols()) * alpha_;
    Eigen::MatrixXd Lm = Kmm.llt().matrixL();
    Eigen::MatrixXd Lmi = Lm.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(Lm.rows(), Lm.cols()));
    Eigen::MatrixXd Kmmi = Lm.transpose().triangularView<Eigen::Upper>().solve(
            Lm.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(Lm.rows(), Lm.cols())));
    
    // compute beta_star
    Eigen::MatrixXd LiUT = Lmi * Knm.transpose();
    Eigen::MatrixXd beta_star = 1.0 / (Knn.array() + sigma_n - LiUT.colwise().squaredNorm().transpose().array());// - LiUT.colwise().squaredNorm().array();Knn.array() + sigma_n
    Eigen::MatrixXd beta_star_sqrt = LiUT.array() * beta_star.array().sqrt().transpose().replicate(LiUT.rows(), 1);
    Eigen::MatrixXd A = beta_star_sqrt * beta_star_sqrt.transpose() + Eigen::MatrixXd::Identity(Kmm.rows(), Kmm.cols());
    Eigen::MatrixXd LA = A.llt().matrixL();

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

gpr_results SparseGPR::predict(const Eigen::MatrixXd & X_test, const bool & return_cov)
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
            results_.y_cov = results_.y_cov.array() * y_train_std_(0) * y_train_std_(0); // Only for 1D output.
            // In sklearn, for 2D output, the K_trans is the same, the y_cov is just [y_cov * std_1, y_cov * std_2]
        }

        // std::cout << "K_trans:\n" << K_trans << std::endl;
        // std::cout << "y_mean:\n" << results_.y_mean << std::endl;
        // std::cout << "V:\n" << V << std::endl;
        // std::cout << "cov:\n" << results_.y_cov << std::endl;
        return results_;
    }
}

SparseGPR::~SparseGPR()
{

}

} // namespace GPRcpp