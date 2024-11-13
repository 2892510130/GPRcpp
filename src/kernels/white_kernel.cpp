#include "kernels/white_kernel.h"

namespace GPRcpp
{
    white_kernel::white_kernel(const double noise_value)
    {
        params_.resize(1);
        params_[0] = noise_value;
    }

    Eigen::MatrixXd white_kernel::k_diag(const Eigen::MatrixXd & x1) const
    {
        return Eigen::MatrixXd::Ones(x1.rows(), 1) * params_[0];
    }

    Eigen::MatrixXd white_kernel::evaluate(const Eigen::MatrixXd & x1) const
    {
        Eigen::MatrixXd k = Eigen::MatrixXd::Identity(x1.rows(), x1.rows());
        return k * params_[0];
    }

    Eigen::MatrixXd white_kernel::evaluate(const Eigen::MatrixXd & x1, const Eigen::MatrixXd & x2) const
    {
        Eigen::MatrixXd k = Eigen::MatrixXd::Zero(x1.rows(), x2.rows());
        return k;
    }

    double white_kernel::test()
    {
        return 1.1;
    }
} // namespace GPRcpp