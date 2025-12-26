#pragma once
#include "kernels/sum_kernel.h"
#include "kernels/product_kernel.h"
#include "gpr_math/distance.h"
#include <fstream>

namespace GPRcpp
{
struct gpr_results
{
    Eigen::VectorXd y_mean;
    Eigen::MatrixXd y_cov;
    Eigen::MatrixXd y_covariance;
    Eigen::MatrixXd dmu_dx;
};


class gpr
{
public:

    gpr();

    gpr(std::shared_ptr<kernel_base> kernel);

    gpr(std::shared_ptr<kernel_base> kernel, bool normalize_y);

    ~gpr();

    virtual void fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train) = 0;
    
    virtual void fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train, const Eigen::MatrixXd & inducing_points) = 0;
    
    virtual gpr_results predict(const Eigen::MatrixXd & X_test) = 0;

    virtual gpr_results predict(const Eigen::MatrixXd & X_test, bool return_cov, bool compute_jac = false) = 0;

    virtual void save_data(const std::string& filename) = 0;

    virtual void load_data(const std::string& filename) = 0;

public:
    std::shared_ptr<kernel_base> kernel_;
    int inference_method = 0;
    double alpha_ = 1e-8; // For the robustness of the llt decomp of the K(X, X) matrix, and act as a noise too.
    double likelihood_varience = 1.0;
    Eigen::MatrixXd L_; // LL^T = K(X, X) for exact gpr
    Eigen::MatrixXd Alpha_; // Alpha_ = K^{-1} * y
    bool has_x_train_;
    bool normalize_y_ = false;
    Eigen::MatrixXd m_X_train;
    Eigen::MatrixXd m_y_train;
    Eigen::RowVectorXd y_train_mean_;
    Eigen::RowVectorXd y_train_std_;
    gpr_results results_;
};

}