#pragma once
#include <vector>
#include <memory>
#include <Eigen/Core>

namespace GPRcpp
{
/* This contains class info of kernel base */
class kernel_base 
{
// This are functions of kernel base
public:
    ~kernel_base() = default;

    std::vector<double> get_params() const;

    void set_params(const std::vector<double> & params);

    virtual Eigen::MatrixXd k_diag(const Eigen::MatrixXd & x1) const = 0;

    virtual Eigen::MatrixXd evaluate(const Eigen::MatrixXd & x1) const = 0;

    virtual Eigen::MatrixXd evaluate(const Eigen::MatrixXd & x1, const Eigen::MatrixXd & x2) const = 0;

protected:
    kernel_base() = default;

// This are variables of kernel base
public:

protected:
    std::vector<double> params_;
};

/* This contains class info of kernel operators */
class kernel_operator : public virtual kernel_base
{
// This are functions of kernel operator
public:
    kernel_operator(std::shared_ptr<kernel_base> left, std::shared_ptr<kernel_base> right);
    kernel_operator() = default;
    ~kernel_operator();
protected:
// This are variables of kernel operator
public:
    std::shared_ptr<kernel_base> left_;
    std::shared_ptr<kernel_base> right_;
};

} // namespace GPRcpp