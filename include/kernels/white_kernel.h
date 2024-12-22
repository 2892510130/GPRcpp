#pragma once
#include "kernels.h"

namespace GPRcpp
{

class white_kernel : public kernel_base
{
public:
    white_kernel(const double noise_value);
    Eigen::MatrixXd k_diag(const Eigen::MatrixXd & x1) const override;
    Eigen::MatrixXd evaluate(const Eigen::MatrixXd & x1) const override;
    Eigen::MatrixXd evaluate(const Eigen::MatrixXd & x1, const Eigen::MatrixXd & x2) const override;
    Eigen::MatrixXd dk_dx(const Eigen::MatrixXd & x1, const Eigen::MatrixXd & x2) const override;

protected:

// This are variables of kernel base
public:

protected:
};

}