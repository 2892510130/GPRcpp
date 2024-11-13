#include "kernels/sum_kernel.h"

namespace GPRcpp
{
    
Eigen::MatrixXd sum_kernel::evaluate(const Eigen::MatrixXd & x1, const Eigen::MatrixXd & x2) const
{
    Eigen::MatrixXd sum_ = left_->evaluate(x1, x2) + right_->evaluate(x1, x2);
    return sum_;
}

Eigen::MatrixXd sum_kernel::evaluate(const Eigen::MatrixXd & x1) const
{
    Eigen::MatrixXd sum_ = left_->evaluate(x1) + right_->evaluate(x1);
    return sum_;
}

Eigen::MatrixXd sum_kernel::k_diag(const Eigen::MatrixXd & x1) const
{
    Eigen::MatrixXd sum_ = left_->k_diag(x1) + right_->k_diag(x1);
    return sum_;
}

}