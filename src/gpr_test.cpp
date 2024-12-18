#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "gprs/sparse_gpr.h"
#include "gprs/exact_gpr.h"
#include "utils/file_operation.h"
#include <chrono>

using namespace GPRcpp;

void test_with_minimal_data()
{
    /* In varDTC
        mu:
        [[-0.28384087]
        [ 0.64530795]
        [ 0.27753041]]
        cov:
        [[0.68861111 0.08581716 0.01170406]
        [0.08581716 0.61724472 0.25306279]
        [0.01170406 0.25306279 2.20023659]]

       In FITC
        mu:
        [[-0.2942645 ]
        [ 0.41992537]
        [ 0.1844007 ]]
        cov:
        [[0.68871227 0.08797517 0.01259572]
        [0.08797517 0.66390237 0.27234206]
        [0.01259572 0.27234206 2.20820292]]
    */
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

    SparseGPR spgp(realkernelPtr, false);
    spgp.alpha_ = 1e-8;
    spgp.use_ldlt_ = true;
    spgp.inference_method = 1;
    spgp.fit(data.m_feature, data.m_output.col(0), data.m_inducing_points);
    auto result = spgp.predict(data.m_feature, true);

    std::cout << "\nmean:\n" << result.y_mean << std::endl;
    std::cout << "\ncov:\n" << result.y_cov << std::endl;
}

void test_with_big_data()
{
    std::string file_path = "../Log/test.txt";
    GPData data = read_sparse_gp_data_from_file(file_path, 12, 2, 520, 40, true);
    Eigen::RowVectorXd ard_length_scale_ = Eigen::RowVectorXd(12);
    ard_length_scale_ << 2776.7324501710436, 3.247155587332236, 1954.5696329279183, 8.844556482918314, 3113.085020209924, 2649.800377250457, 2918.731746800648, 3090.7772548171697, 3149.4249791237094, 1994.60759481792, 3131.4280971043454, 3.6782262654484428;
    std::shared_ptr<kernel_base> constant_kernel_ptr_1 = std::make_shared<constant_kernel>(7891.7937052127345);
    std::shared_ptr<kernel_base> constant_kernel_ptr_2 = std::make_shared<constant_kernel>(5.562684646268137e-309);
    std::shared_ptr<kernel_base> rbf_kernel_ptr_1 = std::make_shared<rbf_kernel>(ard_length_scale_);
    std::shared_ptr<kernel_base> rbf_kernel_ptr_2 = std::make_shared<rbf_kernel>(7.657695755181217);
    std::shared_ptr<kernel_base> realkernelPtr_1 = std::make_shared<product_kernel>(constant_kernel_ptr_1, rbf_kernel_ptr_1);
    std::shared_ptr<kernel_base> realkernelPtr_2 = std::make_shared<product_kernel>(constant_kernel_ptr_2, rbf_kernel_ptr_2);
    std::shared_ptr<kernel_base> realkernelPtr = std::make_shared<sum_kernel>(realkernelPtr_1, realkernelPtr_2);
    SparseGPR spgp(realkernelPtr, false);
    spgp.alpha_ = 1e-8;
    // 0.27325828172487143, 5.562684646268137e-309, 7.657695755181217, 7891.7937052127345, 2776.7324501710436, 3.247155587332236, 1954.5696329279183, 8.844556482918314, 3113.085020209924, 2649.800377250457, 2918.731746800648, 3090.7772548171697, 3149.4249791237094, 1994.60759481792, 3131.4280971043454, 3.6782262654484428
    spgp.likelihood_varience = 0.27325828172487143;
    spgp.inference_method = 0;
    spgp.fit(data.m_feature, data.m_output.col(1), data.m_inducing_points_additional);
    auto result = spgp.predict(data.m_feature, true);

    std::cout << "\nmean:\n" << result.y_mean.block(0, 0, 16, 1) << std::endl;
    std::cout << "\ncov:\n" << result.y_cov.block(0, 0, 4, 4) << std::endl;
}

int main(int argc, char *argv[]) 
{
    if (atoi(argv[1]) == 0)
    {
        test_with_minimal_data();
    }
    else
    {
        test_with_big_data();
    }

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