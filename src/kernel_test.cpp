#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <vector>
#include "gprs/exact_gpr.h"

using namespace GPRcpp;

int main(int argc, char *argv[]) 
{
    double precision_request = 1e-5;

    // Input and output test data
    std::cout << "\n-----Input and output test data-----" << std::endl;
    Eigen::MatrixXd x = Eigen::MatrixXd(3, 4);
    Eigen::MatrixXd y = Eigen::MatrixXd(3, 1);
    Eigen::RowVectorXd ard_length_scale_ = Eigen::RowVectorXd(4);
    x << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2;
    y << -0.5, 0.5, 1.5;
    ard_length_scale_ << 1, 2, 3, 4;
    Eigen::MatrixXd new_x = x.array() / ard_length_scale_.replicate(x.rows(), 1).array();
    std::cout << "x:\n" << x << std::endl;
    std::cout << "y:\n" << y << std::endl;
    std::cout << "ard length scale:\n" << ard_length_scale_ << std::endl;
    std::cout << "new x:\n" << new_x << std::endl;

    // Constant kernel test
    std::cout << "\n-----Constant kernel test-----" << std::endl;
    constant_kernel kernel(0.2);
    Eigen::MatrixXd z = kernel.evaluate(x);
    std::cout << "constant kernel with x:\n" << z << std::endl;
    z = kernel.evaluate(x, y);
    std::cout << "constant kernel with x and y:\n" << z << std::endl;

    // White kernel test
    std::cout << "\n-----White kernel test-----" << std::endl;
    white_kernel kernel2(0.3);
    z = kernel2.evaluate(x, y);
    std::cout << "white kernel with x and y:\n" << z << std::endl;
    z = kernel2.evaluate(x);
    std::cout << "white kernel without y:\n" << z << std::endl;

    // Distances test
    std::cout << "\n-----Distances test-----" << std::endl;
    auto dist = distance_calculator(distance_type::euclidean);
    auto d = dist.compute(x / 0.5);
    std::cout << "distance of euclidean:\n" << pow(d.array(), 2) << std::endl;
    d = exp(-0.5 * d.array());
    std::cout << "distance of euclidean after exp:\n" << d << std::endl;
    d = dist.compute_2d(x, x);
    std::cout << "distance of euclidean of two sample:\n" << d << std::endl;

    // RBF kernel test
    std::cout << "\n-----RBF kernel test-----" << std::endl;
    rbf_kernel kernel3(ard_length_scale_);
    z = kernel3.evaluate(x);
    std::cout << "rbf kernel with x:\n" << z << std::endl;
    Eigen::MatrixXd x_new = x.array() * 2;
    z = kernel3.evaluate(x, x_new);
    std::cout << "rbf kernel with (x,x):\n" << z << std::endl;

    // Combanition of kernels test
    std::cout << "\n-----Combanition of kernels test-----" << std::endl;
    Eigen::MatrixXd comb = kernel3.evaluate(x).array() * kernel.evaluate(x).array();
    std::cout << "comb kernel:\n" << comb << std::endl;
    Eigen::MatrixXd comb2 = comb + kernel2.evaluate(x);
    std::cout << "comb2 kernel:\n" << comb2 << std::endl;

    // Kernel Operators
    std::cout << "\n-----Kernel Operators-----" << std::endl;
    std::shared_ptr<kernel_base> kernelPtr = std::make_shared<constant_kernel>(kernel);
    std::shared_ptr<kernel_base> kernel2Ptr = std::make_shared<white_kernel>(kernel2);
    std::shared_ptr<kernel_base> kernel3Ptr = std::make_shared<rbf_kernel>(kernel3);
    sum_kernel sum_kernel_(kernel3Ptr, kernel2Ptr);
    std::cout << "sum kernel params size:\n" <<  sum_kernel_.get_params().size() << std::endl;
    z = sum_kernel_.evaluate(x);
    std::cout << "sum kernel:\n" << z << std::endl;

    product_kernel product_kernel_(kernelPtr, kernel3Ptr);
    std::cout << "product kernel params size:\n" <<  product_kernel_.get_params().size() << std::endl;
    z = product_kernel_.evaluate(x);
    std::cout << "product kernel:\n" << z << std::endl;

    std::shared_ptr<kernel_base> product_kernel_ptr = std::make_shared<product_kernel>(product_kernel_);
    sum_kernel real_kernel(product_kernel_ptr, kernel2Ptr);
    std::cout << "real kernel params size:\n" <<  real_kernel.get_params().size() << std::endl;
    z = real_kernel.evaluate(x);
    std::cout << "real kernel:\n" << z << std::endl;

    // GPR fit test
    std::cout << "\n-----GPR fit test-----" << std::endl;
    std::shared_ptr<kernel_base> realkernelPtr = std::make_shared<sum_kernel>(real_kernel);
    ExactGPR gpr_obj(realkernelPtr);
    std::cout << "gpr kernel:\n" << gpr_obj.kernel_->evaluate(x) << std::endl;
    gpr_obj.fit(x, y);

    // GPR predict test
    std::cout << "\n-----GPR predict test-----" << std::endl;
    auto predict_ = gpr_obj.predict(x, true);
    std::cout << "gpr predict mean:\n" << predict_.y_mean.transpose() << std::endl;
    std::cout << "gpr predict cov:\n" << predict_.y_cov << std::endl;
    gpr_results gpr_results_obj_;
    gpr_results_obj_.y_mean.resize(3);
    gpr_results_obj_.y_cov.resize(3, 3);
    gpr_results_obj_.y_mean << 0.110755, 0.337883, 0.502956;
    auto error_predict = (gpr_results_obj_.y_mean - predict_.y_mean).norm();
    auto is_equal_predict = error_predict < precision_request;
    gpr_results_obj_.y_cov << 0.38930889, 0.0643255, 0.0304788, 0.0643255, 0.37407895, 0.0643255, 0.0304788, 0.0643255, 0.38930889;
    auto error_predict_cov = (gpr_results_obj_.y_cov - predict_.y_cov).norm();
    auto is_equal_predict_cov = error_predict_cov < precision_request;
    if (!is_equal_predict || !is_equal_predict_cov) {
        std::cout << "[Error]: check the python scripts for reference!!!" << std::endl;
        std::cout << "gpr predict error in mean: " << error_predict << std::endl;
        std::cout << "gpr predict error in cov: " << error_predict_cov << std::endl;
    }

    // Test normalization
    std::cout << "\n-----GPR normalized fit test-----" << std::endl;
    Eigen::MatrixXd y_multi = Eigen::MatrixXd(3, 2);
    y_multi << 1, 2, 3, 4, 5, 6; // The normalized y_train mean and std are correct with this y_train
    ExactGPR gpr_normalized(realkernelPtr, true);
    gpr_normalized.fit(x, y);
    auto predict_normalized = gpr_normalized.predict(x, true);
    std::cout << "gpr predict mean:\n" << predict_normalized.y_mean.transpose() << std::endl;
    std::cout << "gpr predict cov:\n" << predict_normalized.y_cov << std::endl;

    // Change the alpha
    std::cout << "\n-----Change the alpha test-----" << std::endl;
    ExactGPR gpr_change_alpha(realkernelPtr, true);
    gpr_change_alpha.alpha_ = 1e-1;
    gpr_change_alpha.fit(x, y);
    predict_normalized = gpr_change_alpha.predict(x, true);
    std::cout << "gpr predict mean:\n" << predict_normalized.y_mean.transpose() << std::endl;
    std::cout << "gpr predict cov:\n" << predict_normalized.y_cov << std::endl;


    return 0;
}