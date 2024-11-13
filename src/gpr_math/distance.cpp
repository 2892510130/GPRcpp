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
        size_t n = x1.rows();
        size_t dist_size = n * (n - 1) / 2;
        Eigen::MatrixXd dist_(dist_size, 1);
        int index_ = 0;

        if (distance_type_ == distance_type::euclidean)
        {
            for (int i = 0; i < n - 1; ++i)
            {
                for (int j = i + 1; j < n; ++j)
                {
                    dist_(index_) = (x1.row(i) - x1.row(j)).squaredNorm();
                    ++index_;
                }
            }
        }

        return dist_;
    }

    Eigen::MatrixXd distance_calculator::compute_2d(const Eigen::MatrixXd &x1, const Eigen::MatrixXd &x2) const
    {
        size_t n1 = x1.rows();
        size_t n2 = x2.rows();
        Eigen::MatrixXd dist_(n1, n2);

        if (distance_type_ == distance_type::euclidean)
        {
            for (int i = 0; i < n1; i++)
            {
                for (int j = 0; j < n2; j++)
                {
                    dist_(i, j) = (x1.row(i) - x2.row(j)).squaredNorm();
                }
            }
        }

        // Another way to wirte it is uing replicate, but the memory problem is big

        return dist_;
    }
} // namespace GPRcpp