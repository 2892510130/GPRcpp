#include "kernels/rbf_kernel.h"
#include "gpr_math/distance.h"
#include <iostream>

namespace GPRcpp
{
    rbf_kernel::rbf_kernel(const Eigen::RowVectorXd & length_scale)
    {
        params_.resize(length_scale.cols());

        // This can not work before Eigen 3.4
        // int i = 0;
        // for(auto l : length_scale)
        // {
        //     params_[i] = l;
        //     i += 1;
        // }

        for(int i = 0; i < length_scale.size(); i++)
        {
            params_[i] = length_scale(i);
        }

        m_length_scale = length_scale;
        m_ard = true;
    }

    rbf_kernel::rbf_kernel(double length_scale)
    {
        params_.resize(1);
        params_[0] = length_scale;
    }

    Eigen::MatrixXd rbf_kernel::k_diag(const Eigen::MatrixXd & x1) const
    {
        return Eigen::MatrixXd::Ones(x1.rows(), 1);
    }

    Eigen::MatrixXd rbf_kernel::evaluate(const Eigen::MatrixXd & x1) const
    {
        auto dist_calc = distance_calculator(distance_type::euclidean);
        // std::cout << "x1:\n" << x1 << std::endl;
        // std::cout << "params_:\n" << params_[0] << std::endl;
        Eigen::MatrixXd dist;
        if (m_ard) dist = dist_calc.compute(x1.array() / m_length_scale.replicate(x1.rows(), 1).array());
        else dist = dist_calc.compute(x1 / params_[0]);

        dist = exp(-0.5 * dist.array());
        Eigen::MatrixXd k = squareform(dist, x1.rows());
        return k + Eigen::MatrixXd::Identity(x1.rows(), x1.rows());
    }

    Eigen::MatrixXd rbf_kernel::evaluate(const Eigen::MatrixXd & x1, const Eigen::MatrixXd & x2) const
    {
        auto dist_calc = distance_calculator(distance_type::euclidean);
        Eigen::MatrixXd dist;
        if (m_ard) dist = dist_calc.compute_2d(x1.array() / m_length_scale.replicate(x1.rows(), 1).array(), x2.array() / m_length_scale.replicate(x2.rows(), 1).array());
        else dist = dist_calc.compute_2d(x1 / params_[0], x2 / params_[0]);

        dist = exp(-0.5 * dist.array());
        return dist;
    }

    Eigen::MatrixXd rbf_kernel::squareform(const Eigen::MatrixXd & x, size_t size_) const
    {
        Eigen::MatrixXd k = Eigen::MatrixXd::Zero(size_, size_);

        int i = 1;
        int j = 0;
        int count = 0;
        while (count < x.rows())
        {
            if (i == size_)
            {
                j++;
                i = j + 1;
            }

            k(i, j) = x(count, 0);
            k(j, i) = x(count, 0);
            i++;
            count++;
        }
        return k;
    }

    Eigen::MatrixXd rbf_kernel::dk_dx(const Eigen::MatrixXd & x1, const Eigen::MatrixXd & x2) const
    {
        Eigen::MatrixXd dk_dl_left = (x1 - x2.replicate(x1.rows(), 1)).array() / m_length_scale.replicate(x1.rows(), 1).array().square();

        // std::cout << "History data:" << std::endl;
        // std::cout << x1 << std::endl;
        // std::cout << "New data:" << std::endl;
        // std::cout << x2 << std::endl;
        // std::cout << "dk_dl without lengthscale" << std::endl;
        // std::cout << x1 - x2.replicate(x1.rows(), 1) << std::endl;
        // std::cout << "Length scale:" << std::endl;
        // std::cout << m_length_scale << std::endl;
        // std::cout << "dk_dl_left" << std::endl;
        // std::cout << dk_dl_left << std::endl;
        // std::cout << "dk_dl_right" << std::endl;
        // std::cout << evaluate(x1, x2).replicate(1, x2.cols()).array() << std::endl;
        // std::cout << "dk_dl" << std::endl;
        // std::cout << dk_dl_left.array() * evaluate(x1, x2).replicate(1, x2.cols()).array() << std::endl;

        return dk_dl_left.array() * evaluate(x1, x2).replicate(1, x2.cols()).array();
    }
} // namespace GPRcpp