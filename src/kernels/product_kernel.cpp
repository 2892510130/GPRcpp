#include "kernels/product_kernel.h"

namespace GPRcpp
{
    
Eigen::MatrixXd product_kernel::evaluate(const Eigen::MatrixXd & x1, const Eigen::MatrixXd & x2) const
{
    Eigen::MatrixXd product = left_->evaluate(x1, x2).array() * right_->evaluate(x1, x2).array();
    return product;
}

Eigen::MatrixXd product_kernel::evaluate(const Eigen::MatrixXd & x1) const
{
    Eigen::MatrixXd product;
    product.noalias() = (left_->evaluate(x1).array() * right_->evaluate(x1).array()).matrix();
    return product;
}

Eigen::MatrixXd product_kernel::k_diag(const Eigen::MatrixXd & x1) const
{
    Eigen::MatrixXd product = left_->k_diag(x1).array() * right_->k_diag(x1).array();
    return product;
}

Eigen::MatrixXd product_kernel::dk_dx(const Eigen::MatrixXd & x1, const Eigen::MatrixXd & x2) const
{
    return left_->dk_dx(x1, x2).array() * right_->evaluate(x1, x2).replicate(1, x2.cols()).array() + left_->evaluate(x1, x2).replicate(1, x2.cols()).array() * right_->dk_dx(x1, x2).array();
}

}