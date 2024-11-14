#include <iostream>
#include <Eigen/Core>
#include <vector>
#include "gprs/sparse_gpr.h"
#include "utils/file_operation.h"
#include <chrono>

using namespace GPRcpp;

int main(int argc, char *argv[]) 
{
    std::string file_path = "../Log/test_util.txt";
    GPData data = read_sparse_gp_data_from_file(file_path, 4, 2, 3, 2, true);
    std::cout << data.m_feature << std::endl;
    std::cout << data.m_output << std::endl;
    std::cout << data.m_inducing_points << std::endl;
    std::cout << data.m_inducing_points_additional << std::endl;
    Eigen::RowVectorXd ard_length_scale_ = Eigen::RowVectorXd(4);
    ard_length_scale_ << 1, 2, 3, 4;
    double ard_length_scale_2 = 0.5;
    std::shared_ptr<kernel_base> constant_kernel_ptr = std::make_shared<constant_kernel>(0.5);
    std::shared_ptr<kernel_base> constant_kernel_ptr_2 = std::make_shared<constant_kernel>(2);
    std::shared_ptr<kernel_base> rbf_kernel_ptr = std::make_shared<rbf_kernel>(ard_length_scale_);
    std::shared_ptr<kernel_base> rbf_kernel_ptr_2 = std::make_shared<rbf_kernel>(ard_length_scale_2);
    std::shared_ptr<kernel_base> realkernelPtr_1 = std::make_shared<product_kernel>(constant_kernel_ptr, rbf_kernel_ptr);
    std::shared_ptr<kernel_base> realkernelPtr_2 = std::make_shared<product_kernel>(constant_kernel_ptr_2, rbf_kernel_ptr_2);
    std::shared_ptr<kernel_base> realkernelPtr = std::make_shared<sum_kernel>(realkernelPtr_1, realkernelPtr_2);

    // SparseGPR spgp(realkernelPtr_3, false);
    // spgp.alpha_ = 1e-8;
    // spgp.fit(data.m_feature, data.m_output.col(0), data.m_inducing_points);
    // auto result = spgp.predict(data.m_feature, true);

    // std::string file_path = "../Log/test_sparse.txt";
    // GPData data = read_sparse_gp_data_from_file(file_path, 12, 2, 780, 40);
    // Eigen::RowVectorXd ard_length_scale_ = Eigen::RowVectorXd(12);
    // ard_length_scale_ << 1e+02, 0.0942, 0.1, 0.0772, 0.194, 0.156, 1e+02, 1e+02, 0.0257, 0.059, 0.0925, 0.07;
    // std::shared_ptr<kernel_base> constant_kernel_ptr_1 = std::make_shared<constant_kernel>(3.08*3.08);
    // std::shared_ptr<kernel_base> constant_kernel_ptr_2 = std::make_shared<constant_kernel>(0.204*0.204);
    // std::shared_ptr<kernel_base> rbf_kernel_ptr_1 = std::make_shared<rbf_kernel>(ard_length_scale_);
    // std::shared_ptr<kernel_base> rbf_kernel_ptr_2 = std::make_shared<rbf_kernel>(0.0037);
    // std::shared_ptr<kernel_base> realkernelPtr_1 = std::make_shared<product_kernel>(constant_kernel_ptr_1, rbf_kernel_ptr_1);
    // std::shared_ptr<kernel_base> realkernelPtr_2 = std::make_shared<product_kernel>(constant_kernel_ptr_2, rbf_kernel_ptr_2);
    // std::shared_ptr<kernel_base> realkernelPtr = std::make_shared<sum_kernel>(realkernelPtr_1, realkernelPtr_2);
    SparseGPR spgp(realkernelPtr, false);
    spgp.alpha_ = 1e-8;
    spgp.reference_method = 1;
    spgp.fit(data.m_feature, data.m_output.col(0), data.m_inducing_points);
    auto result = spgp.predict(data.m_feature, true);

    std::cout << data.m_output.col(0) << std::endl;
    // std::cout << constant_kernel_ptr->k_diag(data.m_feature) << std::endl;
    // std::cout << realkernelPtr_1->k_diag(data.m_feature) << std::endl;
    std::cout << "\nmean:\n" << result.y_mean << std::endl;
    std::cout << "\ncov:\n" << result.y_cov << std::endl;
    
    // Try to optimize the calculate of the dist function, but no progress
    // auto start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < 10; i++) rbf_kernel_ptr.get()->evaluate(data.m_feature, data.m_output);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // std::cout << "Duration is: " << duration << "ms." << std::endl;

    return 0;
}

/*
Test data for the sparse var DTC method is:
0.1 0.2 0.3 0.4 -0.5 1
0.5 0.6 0.7 0.8 0.5 2
0.9 1.0 1.1 1.2 1.5 3
0.1 0.2 0.3 0.4
0.5 0.6 0.7 0.8
*/