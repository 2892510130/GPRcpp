#pragma once
#include "kernels.h"

namespace GPRcpp
{

class rbf_kernel : public kernel_base
{
public:
    rbf_kernel(const Eigen::RowVectorXd & length_scale);
    rbf_kernel(double length_scale);
    Eigen::MatrixXd k_diag(const Eigen::MatrixXd & x1) const override;
    Eigen::MatrixXd evaluate(const Eigen::MatrixXd & x1) const override;
    Eigen::MatrixXd evaluate(const Eigen::MatrixXd & x1, const Eigen::MatrixXd & x2) const override;
    Eigen::MatrixXd squareform(const Eigen::MatrixXd & x, size_t size_) const;

protected:

// This are variables of kernel base
public:
    Eigen::RowVectorXd m_length_scale;
    bool m_ard = false;
protected:
};

}