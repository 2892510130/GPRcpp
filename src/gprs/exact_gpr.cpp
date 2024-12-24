#include "gprs/exact_gpr.h"
#include <iostream>
#include <Eigen/Dense>

namespace GPRcpp
{

void ExactGPR::fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train, const Eigen::MatrixXd & inducing_points)
{
    std::cout << "<!!!! It is exact GPR, please use fit function without inducing points !!!!>" << std::endl;
    throw std::runtime_error("Not implemented");
}

void ExactGPR::fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train)
{
    std::cout << "<***** Fitting GPR *****>" << std::endl;
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

    if (use_ldlt_)
    {
        Eigen::LDLT<Eigen::MatrixXd> Lm_LLT = K.ldlt();
        Eigen::MatrixXd L1 = Lm_LLT.matrixL();
        Eigen::MatrixXd D1 = Lm_LLT.vectorD().cwiseSqrt();  // D 的平方根
        Eigen::MatrixXd L_ = Lm_LLT.transpositionsP() *  L1 * D1.asDiagonal().toDenseMatrix();
        Alpha_ = Lm_LLT.solve(y_train_);
    }
    else
    {
        Eigen::LLT<Eigen::MatrixXd> LLT_ = K.llt();
        L_ = LLT_.matrixL();
        Alpha_ = LLT_.solve(y_train_);
    }

    // For debug
    // std::cout << "temp_y decomp:\n" << temp_y.matrix() << std::endl;
    // std::cout << "Alpha_:\n" << Alpha_ << std::endl;
    // std::cout << "direct solver answer of K wrt y_train:\n" << K.llt().solve(y_train_) << std::endl;
    // std::cout << "K * L_ - y_train(should near zero):\n" << K * Alpha_ - y_train << std::endl;
}

gpr_results ExactGPR::predict(const Eigen::MatrixXd & X_test, bool return_cov)
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
        auto K_trans = kernel_->evaluate(X_test, X_train_);
        results_.y_mean = K_trans * Alpha_;

        if (return_cov)
        {
            Eigen::MatrixXd V = L_.triangularView<Eigen::Lower>().solve(K_trans.transpose());
            results_.y_cov = kernel_->evaluate(X_test) - V.transpose() * V;
        }

        if (normalize_y_)
        {
            results_.y_mean = (results_.y_mean.array().rowwise() * y_train_std_.array()).rowwise() + y_train_mean_.array();
            results_.y_cov = results_.y_cov * y_train_std_(0) * y_train_std_(0); // Only for 1D output.
            // In sklearn, for 2D output, the K_trans is the same, the y_cov is just [y_cov * std_1, y_cov * std_2]
        }

        // std::cout << "X_test:\n" << X_test << std::endl;
        // std::cout << "X_train_:\n" << X_train_ << std::endl;
        // std::cout << "K_trans:\n" << K_trans << std::endl;
        // std::cout << "y_mean:\n" << results_.y_mean << std::endl;
        // std::cout << "V:\n" << V << std::endl;
        // std::cout << "cov:\n" << results_.y_cov << std::endl;
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

    const Eigen::MatrixXd dk_dx =  kernel_->dk_dx(X_train_, X_test);

    const Eigen::MatrixXd dmu_dx = dk_dx.transpose() * Alpha_;

    double first_order_varience = (dmu_dx.transpose() * input_cov * dmu_dx)(0);

    gpr_results certain_predict = predict(X_test, true);

    if (add_covariance)
    {
        certain_predict.y_covariance = dmu_dx.transpose() * input_cov;
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

ExactGPR::~ExactGPR()
{

}

}