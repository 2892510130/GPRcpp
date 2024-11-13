#pragma once
#include "kernels.h"
#include "white_kernel.h"
#include "constant_kernel.h"
#include "rbf_kernel.h"

namespace GPRcpp
{
/* This contains class info of kernel operator sum */
class sum_kernel : public kernel_operator
{
// This are functions of sum kernel
public:
  sum_kernel(std::shared_ptr<kernel_base> left, std::shared_ptr<kernel_base> right) : kernel_operator(left, right) {};
  Eigen::MatrixXd k_diag(const Eigen::MatrixXd & x1) const override;
  Eigen::MatrixXd evaluate(const Eigen::MatrixXd & x1) const override;
  Eigen::MatrixXd evaluate(const Eigen::MatrixXd & x1, const Eigen::MatrixXd & x2) const override;
protected:
// This are variables of sum kernel
public:
};
}