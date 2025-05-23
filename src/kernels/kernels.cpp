#include "kernels/kernels.h"
#include <stdexcept>
#include <iostream>

namespace GPRcpp
{
    
std::vector<double> kernel_base::get_params() const
{
    return params_;
}

void kernel_base::set_params(const std::vector<double> & params)
{
    params_ = params;
}

Eigen::MatrixXd kernel_base::evaluate(const Eigen::MatrixXd & x1) const
{
    throw std::runtime_error("Not implemented");
}

kernel_operator::kernel_operator(std::shared_ptr<kernel_base> left, std::shared_ptr<kernel_base> right)
{
    left_ = left;
    right_ = right;
    auto left_params = left_->get_params();
    auto right_params = right_->get_params();
    params_.resize(left_params.size() + right_params.size());
    std::copy(left_params.begin(), left_params.end(), params_.begin());
    std::copy(right_params.begin(), right_params.end(), params_.begin() + left_params.size());
    // std::cout << "params:\n" << params_[0] << '\n';
    // std::cout << "params:\n" << params_[1] << '\n';
}

kernel_operator::~kernel_operator()
{
    // delete left_;
    // delete right_;
}

} // namespace GPRcpp
