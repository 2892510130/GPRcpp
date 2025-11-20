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
    X_train_ = X_train;
    y_train_ = y_train;

    if (normalize_y_)
    {
        y_train_mean_ = y_train_.colwise().mean();
        y_train_std_ = ((y_train_.rowwise() - y_train_mean_).array().square().colwise().sum() / y_train_.rows()).sqrt();
        y_train_ = (y_train_.rowwise() - y_train_mean_).array().rowwise() / y_train_std_.array();
    }

    auto K = kernel_->evaluate(X_train_);
    K += Eigen::MatrixXd::Identity(K.rows(), K.cols()) * alpha_;
    Eigen::LLT<Eigen::MatrixXd> LLT_ = K.llt();
    Alpha_ = LLT_.solve(y_train_);

    // For debug
    // std::cout << "temp_y decomp:\n" << temp_y.matrix() << '\n';
    // std::cout << "Alpha_:\n" << Alpha_ << '\n';
    // std::cout << "direct solver answer of K wrt y_train:\n" << K.llt().solve(y_train_) << '\n';
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
        auto K_trans = kernel_->evaluate(X_test, X_train_);
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
        // std::cout << "X_train_:\n" << X_train_ << '\n';
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

gpr_results ExactGPR::predict_at_uncertain_input(const Eigen::MatrixXd & X_test, const Eigen::MatrixXd & input_cov)
{
    return predict_at_uncertain_input(X_test, input_cov, false, false);
}

gpr_results ExactGPR::predict_at_uncertain_input(const Eigen::MatrixXd & X_test, const Eigen::MatrixXd & input_cov, bool add_covariance, bool add_second_order_variance)
{
    if (X_test.rows() != 1) throw std::runtime_error("ExactGPR::predict_at_uncertain_input only support 1 sample input!");

    gpr_results certain_predict = predict(X_test, true);

    const Eigen::MatrixXd dk_dx =  kernel_->dk_dx(X_train_, X_test);

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

    // std::cout << "[ExactGPR]: dmu_dx.T * input_cov is:\n" << certain_predict.y_covariance << '\n';
    // std::cout << "[ExactGPR]: dk_dx is:\n" << dk_dx << '\n';
    // std::cout << "[ExactGPR]: Alpha_ is:\n" << Alpha_ << '\n';
    // std::cout << "[ExactGPR]: dmu_dx is:\n" << dmu_dx << '\n';
    // std::cout << "[ExactGPR]: first_order_varience is:\n" << first_order_varience << '\n';
    // std::cout << "[ExactGPR]: exact cov is:\n" << certain_predict.y_cov << '\n';

    return certain_predict;
}

ExactGPR::~ExactGPR()
{

}

}
