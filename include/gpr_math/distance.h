#pragma once
#include <vector>
#include <Eigen/Core>

namespace GPRcpp
{

enum class distance_type
{
    euclidean,
};

class distance_calculator 
{
// This are functions of kernel base
public:
    ~distance_calculator() = default;

    distance_calculator(distance_type d_type);

    Eigen::MatrixXd compute(const Eigen::MatrixXd &x1) const;

    Eigen::MatrixXd compute_2d(const Eigen::MatrixXd &x1, const Eigen::MatrixXd &x2) const;

protected:

// This are variables of kernel base
public:

protected:
  distance_type distance_type_;
};

} // namespace GPRcpp