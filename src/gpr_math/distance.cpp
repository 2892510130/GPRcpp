#include "gpr_math/distance.h"
#include <iostream>

namespace GPRcpp
{
    distance_calculator::distance_calculator(distance_type d_type)
    {
        distance_type_ = d_type;
    }

    Eigen::MatrixXd distance_calculator::compute(const Eigen::MatrixXd &x1) const
    {
        Eigen::MatrixXd r2;
        if (distance_type_ == distance_type::euclidean)
        {
            Eigen::VectorXd Xsq = x1.rowwise().squaredNorm();

            // Pairwise squared distances
            r2 = (-2.0 * (x1 * x1.transpose())).eval();

            r2.colwise() += Xsq;
            r2.rowwise() += Xsq.transpose();

            // Numerical stability
            r2 = r2.cwiseMax(0.0);

            // Force diagonal to zero
            r2.diagonal().setZero();
        }
        return r2;
    }

    Eigen::MatrixXd distance_calculator::compute_2d(const Eigen::MatrixXd &x1, const Eigen::MatrixXd &x2) const
    {
        Eigen::MatrixXd r2;
        if (distance_type_ == distance_type::euclidean)
        {
            Eigen::VectorXd X1sq = x1.rowwise().squaredNorm();
            Eigen::VectorXd X2sq = x2.rowwise().squaredNorm();

            r2 = (-2.0 * (x1 * x2.transpose())).eval();

            r2.colwise() += X1sq;
            r2.rowwise() += X2sq.transpose();

            r2 = r2.cwiseMax(0.0);
        }
        return r2;
    }
} // namespace GPRcpp