#pragma once
#include "gprs/gpr.h"

namespace GPRcpp
{

class ExactGPR: public gpr
{
public:
    using gpr::gpr;
    ~ExactGPR();
    void fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train) override;
    void fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train, const Eigen::MatrixXd & inducing_points) override;
    gpr_results predict(const Eigen::MatrixXd & X_test) override;
    gpr_results predict(const Eigen::MatrixXd & X_test, const bool & return_cov) override;

public:
    Eigen::MatrixXd m_inducing_point;
    Eigen::MatrixXd woodbury_inv;
};

} // namespace GPRcpp