#pragma once
#include "kernels/sum_kernel.h"
#include "kernels/product_kernel.h"
#include "gpr_math/distance.h"

namespace GPRcpp
{
struct gpr_results
{
    Eigen::VectorXd y_mean;
    Eigen::MatrixXd y_cov;
};


class gpr
{
public:

    gpr();

    gpr(std::shared_ptr<kernel_base> kernel);

    gpr(std::shared_ptr<kernel_base> kernel, bool normalize_y);

    ~gpr();

    void fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train);

    gpr_results predict(const Eigen::MatrixXd & X_test);

    gpr_results predict(const Eigen::MatrixXd & X_test, const bool & return_cov);

public:
    std::shared_ptr<kernel_base> kernel_;
    double alpha_ = 1e-10;
    double beta_;
    Eigen::MatrixXd L_;
    Eigen::MatrixXd Alpha_;
    bool has_x_train_;
    bool normalize_y_ = false;
    Eigen::MatrixXd X_train_;
    Eigen::MatrixXd y_train_;
    Eigen::RowVectorXd y_train_mean_;
    Eigen::RowVectorXd y_train_std_;
    gpr_results results_;
};

}