#include "gprs/variational_gpr.h"
#include <iostream>
#include <Eigen/Dense>

namespace GPRcpp
{
    VariationalGPR::~VariationalGPR()
    {

    }

    void VariationalGPR::fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train, const Eigen::MatrixXd & inducing_points)
    {
        std::cout << "[GPR INFO]: fitting variational GPR with method " << inference_method << '\n';
        has_x_train_ = true;
        X_train_ = X_train;
        y_train_ = y_train;
        m_inducing_point = inducing_points;
        m_N = X_train.rows();
        m_M = m_inducing_point.rows();

        if (normalize_y_)
        {
            y_train_mean_ = y_train_.colwise().mean();
            y_train_std_ = ((y_train_.rowwise() - y_train_mean_).array().square().colwise().sum() / y_train_.rows()).sqrt();
            y_train_ = (y_train_.rowwise() - y_train_mean_).array().rowwise() / y_train_std_.array();
        }

        if (inference_method == 0) fit_with_dtc();
        else if (inference_method == 1) fit_with_fitc();
    }

    void VariationalGPR::fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train)
    {
        std::cout << "<!!!! It is sparse GPR, please use fit function with inducing points !!!!>" << '\n';
        throw std::runtime_error("Not implemented");
    }

    void VariationalGPR::fit_with_fitc()
    {
        Eigen::MatrixXd Kuu = kernel_->evaluate(m_inducing_point);
        Kuu = (Kuu + Kuu.transpose()) / 2.0;
        m_Kuf = kernel_->evaluate(m_inducing_point, X_train_);

        Eigen::MatrixXd jitter = 1e-6 * Eigen::MatrixXd::Identity(m_M, m_M);
        m_Luu = (Kuu + jitter).llt().matrixL();

        m_W = m_Luu.triangularView<Eigen::Lower>().solve(m_Kuf);

        Eigen::MatrixXd Kff_diag = kernel_->k_diag(X_train_).transpose();
        Eigen::MatrixXd Qff_diag = m_W.array().square().colwise().sum();
        m_diag = (Kff_diag - Qff_diag).array() + likelihood_varience;
        m_diag_inv = 1.0 / m_diag.array();
        Eigen::MatrixXd W_diag_inv = m_W.array() * m_diag_inv.array().replicate(m_M, 1);
        m_W_diag_inv_y = W_diag_inv * y_train_;

        Eigen::MatrixXd K = W_diag_inv * m_W.transpose();
        K = (K + K.transpose()) / 2.0 + Eigen::MatrixXd::Identity(m_M, m_M);
        m_L = K.llt().matrixL();

        // Get the mu and Su
        Eigen::MatrixXd tmp_1 = m_L.triangularView<Eigen::Lower>().solve(m_Luu.transpose());
        m_Su = tmp_1.transpose() * tmp_1;
        Eigen::MatrixXd tmp_2 = (m_Kuf.array() * m_diag_inv.array().replicate(m_M, 1)).matrix() * y_train_;
        Eigen::MatrixXd rhs = m_Luu.triangularView<Eigen::Lower>().solve(tmp_2);
        m_mu = m_Luu.triangularView<Eigen::Lower>().solve(m_Su.transpose()).transpose() * rhs;
    }

    void VariationalGPR::fit_with_dtc()
    {

    }

    gpr_results VariationalGPR::predict(const Eigen::MatrixXd & X_test)
    {
        return predict(X_test, false);
    }

    gpr_results VariationalGPR::predict(const Eigen::MatrixXd & X_test, bool return_cov, bool compute_jac)
    {
        Eigen::MatrixXd Kss = kernel_->evaluate(X_test);
        Eigen::MatrixXd Ksu = kernel_->evaluate(X_test, m_inducing_point);
        Eigen::MatrixXd Ws = m_Luu.triangularView<Eigen::Lower>().solve(Ksu.transpose());
        Eigen::MatrixXd tmp = m_Luu.transpose().triangularView<Eigen::Upper>().solve(Ws);
        
        results_.y_mean = Ws.transpose() * m_Luu.triangularView<Eigen::Lower>().solve(m_mu);

        if (return_cov)
            results_.y_cov = Kss - Ws.transpose() * Ws + tmp.transpose() * m_Su * tmp;

        if (compute_jac)
        {
            const Eigen::MatrixXd dk_dx = kernel_->dk_dx(m_inducing_point, X_test);
            // results_.dmu_dx = dk_dx.transpose() * Alpha_; // TODO: implement this
            // if (normalize_y_)
            // {
            //     results_.dmu_dx = (results_.dmu_dx.array().rowwise() * y_train_std_.array());
            // }
        }

        if (normalize_y_)
        {
            results_.y_mean = (results_.y_mean.array().rowwise() * y_train_std_.array()).rowwise() + y_train_mean_.array();
            results_.y_cov = results_.y_cov * y_train_std_(0) * y_train_std_(0); // Only for 1D output.
        }
        
        return results_;
    }

    gpr_results VariationalGPR::predict_at_uncertain_input(const Eigen::MatrixXd & X_test, const Eigen::MatrixXd & input_cov)
    {
        return predict_at_uncertain_input(X_test, input_cov, false, false);
    }

    gpr_results VariationalGPR::predict_at_uncertain_input(const Eigen::MatrixXd & X_test, const Eigen::MatrixXd & input_cov, bool add_covariance, bool add_second_order_variance)
    {
        // TODO: think about this function, does we need it anymore?
        if (X_test.rows() != 1) throw std::runtime_error("VariationalGPR::predict_at_uncertain_input only support 1 sample input!");

        gpr_results certain_predict = predict(X_test, true, true);

        if (add_covariance)
        {
            certain_predict.y_covariance = certain_predict.dmu_dx.transpose() * input_cov;
            if (normalize_y_)
                certain_predict.y_covariance = certain_predict.y_covariance * y_train_std_(0) * y_train_std_(0);
        }

        return certain_predict;
    }
}