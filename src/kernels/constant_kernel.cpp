#include "kernels/constant_kernel.h"

namespace GPRcpp
{
    constant_kernel::constant_kernel(double constant_value)
    {
        params_.resize(1);
        params_[0] = constant_value;
    }

    Eigen::MatrixXd constant_kernel::k_diag(const Eigen::MatrixXd & x1) const
    {
        return Eigen::MatrixXd::Ones(x1.rows(), 1) * params_[0];
    }

    Eigen::MatrixXd constant_kernel::evaluate(const Eigen::MatrixXd & x1) const
    {
        return Eigen::MatrixXd::Constant(x1.rows(), x1.rows(), params_[0]);
    }

    Eigen::MatrixXd constant_kernel::evaluate(const Eigen::MatrixXd & x1, const Eigen::MatrixXd & x2) const
    {
        return Eigen::MatrixXd::Constant(x1.rows(), x2.rows(), params_[0]);
    }

    Eigen::MatrixXd constant_kernel::dk_dx(const Eigen::MatrixXd & x1, const Eigen::MatrixXd & x2) const
    {
        return Eigen::MatrixXd::Zero(x1.rows(), x1.cols());
    }
} // namespace GPRcpp
