#include "gprs/exact_gpr.h"
#include <iostream>
#include <Eigen/Dense>

namespace GPRcpp
{

void ExactGPR::fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train, const Eigen::MatrixXd & inducing_points)
{
    std::cout << "<!!!! It is exact GPR, please use fit function without inducing points !!!!>" << '\n';
    throw std::runtime_error("Not implemented");
}

void ExactGPR::fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train)
{
    std::cout << "<***** Fitting GPR *****>" << '\n';
    has_x_train_ = true;
    m_X_train = X_train;
    m_y_train = y_train;

    if (normalize_y_)
    {
        y_train_mean_ = m_y_train.colwise().mean();
        y_train_std_ = ((m_y_train.rowwise() - y_train_mean_).array().square().colwise().sum() / m_y_train.rows()).sqrt();
        m_y_train = (m_y_train.rowwise() - y_train_mean_).array().rowwise() / y_train_std_.array();
    }

    auto K = kernel_->evaluate(m_X_train);
    K += Eigen::MatrixXd::Identity(K.rows(), K.cols()) * alpha_;
    Eigen::LLT<Eigen::MatrixXd> LLT_ = K.llt();
    Alpha_ = LLT_.solve(m_y_train);

    // For debug
    // std::cout << "temp_y decomp:\n" << temp_y.matrix() << '\n';
    // std::cout << "Alpha_:\n" << Alpha_ << '\n';
    // std::cout << "direct solver answer of K wrt y_train:\n" << K.llt().solve(m_y_train) << '\n';
    // std::cout << "K * L_ - y_train(should near zero):\n" << K * Alpha_ - y_train << '\n';
}

gpr_results ExactGPR::predict(const Eigen::MatrixXd & X_test, bool return_cov, bool compute_jac)
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
        auto K_trans = kernel_->evaluate(X_test, m_X_train);
        results_.y_mean = K_trans * Alpha_;

        if (return_cov)
        {
            Eigen::MatrixXd V;
            V = L_.triangularView<Eigen::Lower>().solve(K_trans.transpose());
            results_.y_cov = kernel_->evaluate(X_test) - V.transpose() * V;
            // std::cout << "V error: " << (L_ * V - K_trans.transpose()).squaredNorm() << '\n'; // V error: 9.45317e-27
        }

        if (normalize_y_)
        {
            results_.y_mean = (results_.y_mean.array().rowwise() * y_train_std_.array()).rowwise() + y_train_mean_.array();
            results_.y_cov = results_.y_cov * y_train_std_(0) * y_train_std_(0); // Only for 1D output.
            // In sklearn, for 2D output, the K_trans is the same, the y_cov is just [y_cov * std_1, y_cov * std_2]
        }

        // std::cout << "X_test:\n" << X_test << '\n';
        // std::cout << "m_X_train:\n" << m_X_train << '\n';
        // std::cout << "K_trans:\n" << K_trans << '\n';
        // std::cout << "y_mean:\n" << results_.y_mean << '\n';
        // std::cout << "V:\n" << V << '\n';
        // std::cout << "cov:\n" << results_.y_cov << '\n';
        return results_;
    }
}

gpr_results ExactGPR::predict(const Eigen::MatrixXd & X_test)
{
    return predict(X_test, false);
}

ExactGPR::~ExactGPR()
{

}

}
