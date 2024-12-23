#pragma once
#include "gprs/gpr.h"

namespace GPRcpp
{

class SparseGPR: public gpr
{
public:
    using gpr::gpr;
    ~SparseGPR();
    void fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train, const Eigen::MatrixXd & inducing_points) override;
    void fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train) override;
    void fit_with_dtc();
    void fit_with_fitc();
    gpr_results predict(const Eigen::MatrixXd & X_test) override;
    gpr_results predict(const Eigen::MatrixXd & X_test, bool return_cov) override;
    gpr_results predict_at_uncertain_input(const Eigen::MatrixXd & X_test, const Eigen::MatrixXd & input_cov) override;
    gpr_results predict_at_uncertain_input(const Eigen::MatrixXd & X_test, const Eigen::MatrixXd & input_cov, bool add_covariance, bool add_second_order_variance) override;

public:
    Eigen::MatrixXd m_inducing_point;
    Eigen::MatrixXd woodbury_inv;
    int sparse_method = 0; // 0 for varDTC, 1 for FITC
};

} // namespace GPRcpp