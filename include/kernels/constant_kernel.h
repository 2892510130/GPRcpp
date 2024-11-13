#pragma once
#include "kernels.h"

namespace GPRcpp
{

class constant_kernel : public kernel_base
{
public:
    constant_kernel(double constant_value);
    Eigen::MatrixXd k_diag(const Eigen::MatrixXd & x1) const override;
    Eigen::MatrixXd evaluate(const Eigen::MatrixXd & x1) const override;
    Eigen::MatrixXd evaluate(const Eigen::MatrixXd & x1, const Eigen::MatrixXd & x2) const override;
    
protected:

// This are variables of kernel base
public:

protected:
};

}