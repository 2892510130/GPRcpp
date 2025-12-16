#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "gprs/sparse_gpr.h"
#include "gprs/variational_gpr.h"
#include "gprs/exact_gpr.h"
#include "utils/file_operation.h"
#include <chrono>
#include <cmath>

using namespace GPRcpp;

void update_cov(Eigen::MatrixXd & cov_before, double new_cov, bool update_covarianceconst, const Eigen::MatrixXd & covariance);
void test_uncertainty_propagation();
void test_uncertainty_propagation_20_times();
void test_with_minimal_data();
void test_with_big_data(int iteration);
void test_for_py_simulator(int sparse_method = 0, bool normalize_gpr = true);
void test_variational_gpr(int sparse_method = 0, bool normalize_gpr = true);
void random_test();

int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        std::cout << "You should select one task!\n";
    }

    if (atoi(argv[1]) == 0)
    {
        std::cout << "You have select minimal data test!" << '\n';
        test_with_minimal_data();
    }
    else if (atoi(argv[1]) == 1)
    {
        std::cout << "You have select big data test!" << '\n';
        test_with_big_data(atoi(argv[2]));
    }
    else if (atoi(argv[1]) == 2)
    {
        std::cout << "You have select uncertainty propagation test!" << '\n';
        test_uncertainty_propagation();
    }
    else if (atoi(argv[1]) == 3)
    {
        std::cout << "You have select uncertainty propagation 20 times test!" << '\n';
        test_uncertainty_propagation_20_times();
    }
	else if (atoi(argv[1]) == 4)
    {
        std::cout << "You have select py simulator test!" << '\n';
        int sparse_method = 0;
        bool normalize_gpr = true;
        if (argc > 2) sparse_method = atoi(argv[2]);
        if (argc > 3) normalize_gpr = true ? atoi(argv[3]) == 0 : false;
        test_for_py_simulator(sparse_method, normalize_gpr);
    }
    else if (atoi(argv[1]) == 5)
    {
        std::cout << "You have select variational gpr test!" << '\n';
        int sparse_method = 0;
        bool normalize_gpr = true;
        if (argc > 2) sparse_method = atoi(argv[2]);
        if (argc > 3) normalize_gpr = true ? atoi(argv[3]) == 0 : false;
        test_variational_gpr(sparse_method, normalize_gpr);
    }
    else if (atoi(argv[1]) == 100)
    {
        std::cout << "You have select random test!" << '\n';
        random_test();
    }

    return 0;
}

void random_test()
{
    std::string file_path = "C:/Users/pc/Desktop/Personal/Code/GPRcpp/Log/py.txt";
    GPData data = read_sparse_gp_data_from_file(file_path, 12, 2, 260, 40, true);

    Eigen::RowVectorXd ard_length_scale_v = Eigen::RowVectorXd(12);
    ard_length_scale_v << 0.8658335733313588, 13.322692325094174, 6.407021934587417, 14.430933792473933, 
                        0.8262508642621557, 12.956010458686743, 14.755220096026246, 17.347816965479478, 
                        2.1033703167736517, 13.100624005598226, 14.435174572545298, 17.816198438786444;
    std::shared_ptr<kernel_base> rbf_kernel_ptr_v = std::make_shared<rbf_kernel>(ard_length_scale_v);
    Eigen::MatrixXd test_1 = data.m_feature.block(0, 0, 4, 12), test_2 = data.m_feature.block(4, 0, 1, 12);
    Eigen::MatrixXd dk_dx = rbf_kernel_ptr_v->dk_dx(test_1, test_2);

    std::cout << "[GPRTest]: dk_dx is\n" << dk_dx << "\n";
}

void test_variational_gpr(int sparse_method, bool normalize_gpr)
{
    std::cout << "[VariationalGPR Test] test with sparse_method " << sparse_method << ", and normalize is " << normalize_gpr << '\n';
    std::string file_path = "C:/Users/pc/Desktop/Personal/Code/GPRcpp/Log/py.txt";
    GPData data = read_sparse_gp_data_from_file(file_path, 12, 2, 260, 40, true);

    // Read for state v
    Eigen::RowVectorXd ard_length_scale_v = Eigen::RowVectorXd(12);
    ard_length_scale_v << 0.8658335733313588, 13.322692325094174, 6.407021934587417, 14.430933792473933, 0.8262508642621557, 12.956010458686743, 14.755220096026246, 17.347816965479478, 2.1033703167736517, 13.100624005598226, 14.435174572545298, 17.816198438786444;
    std::shared_ptr<kernel_base> constant_kernel_ptr_v = std::make_shared<constant_kernel>(15.938946986819923);
    std::shared_ptr<kernel_base> rbf_kernel_ptr_v = std::make_shared<rbf_kernel>(ard_length_scale_v);
	std::shared_ptr<kernel_base> my_kernel_v = std::make_shared<product_kernel>(constant_kernel_ptr_v, rbf_kernel_ptr_v);

    VariationalGPR spgp_v(my_kernel_v, normalize_gpr);
    spgp_v.likelihood_varience = 0.00026610989521133605;
    if (sparse_method != 0) spgp_v.inference_method = 1;
    spgp_v.fit(data.m_feature, data.m_output.col(0), data.m_inducing_points);
	
    // Read for state w
    Eigen::RowVectorXd ard_length_scale_w = Eigen::RowVectorXd(12);
    // ard_length_scale_w << 179.89783828567562, 5.3518386878180575, 278.8128680361318, 37.15662167141493, 171.78692980352864, 176.27457390169434, 272.41892243231155, 258.7789715555678, 184.1016212824687, 5.335272339533333, 287.0791355720346, 220.52174051242275;
    ard_length_scale_w << 0.8658335733313588, 13.322692325094174, 6.407021934587417, 14.430933792473933, 0.8262508642621557, 12.956010458686743, 14.755220096026246, 17.347816965479478, 2.1033703167736517, 13.100624005598226, 14.435174572545298, 17.816198438786444;
    std::shared_ptr<kernel_base> constant_kernel_ptr_w = std::make_shared<constant_kernel>(15.938946986819923);//278.27107082447924
    std::shared_ptr<kernel_base> rbf_kernel_ptr_w = std::make_shared<rbf_kernel>(ard_length_scale_w);
	std::shared_ptr<kernel_base> my_kernel_w = std::make_shared<product_kernel>(constant_kernel_ptr_w, rbf_kernel_ptr_w);

    VariationalGPR spgp_w(my_kernel_w, normalize_gpr);
    spgp_w.likelihood_varience = 0.00026610989521133605;//3.5873462385031854e-05
    if (sparse_method != 0) spgp_w.inference_method = 1;
    spgp_w.fit(data.m_feature, data.m_output.col(1), data.m_inducing_points_additional);

    // Check mean and var
    Eigen::MatrixXd x_test = spgp_v.m_X_train.block(0, 0, 4, spgp_v.m_X_train.cols());
    auto result_v = spgp_v.predict(x_test, true, false);
    auto result_w = spgp_w.predict(x_test, true, false);
    std::cout << "[VariationalGPR Test] Mean result_v:\n" << result_v.y_mean << '\n';
    std::cout << "[VariationalGPR Test] Cov result_v:\n" << result_v.y_cov << '\n';
    std::cout << "[VariationalGPR Test] Mean result_w:\n" << result_w.y_mean << '\n';
    std::cout << "[VariationalGPR Test] Cov result_w:\n" << result_w.y_cov << '\n';
    Eigen::MatrixXd x_test2 = Eigen::MatrixXd::Zero(1, 12);
    x_test2 << 0.00980392,0.0155402,0.192157,   0.304279,     0.0025, 0.00396425,        0.1,    0.15854,          0,          0,          0,          0;
    auto result_v2 = spgp_v.predict(x_test2, true, true);
    auto result_w2 = spgp_w.predict(x_test2, true, true);
    std::cout << "[VariationalGPR Test] gradient result_v 2:\n" << result_v2.dmu_dx.transpose() << '\n';
    std::cout << "[VariationalGPR Test] gradient result_w 2:\n" << result_w2.dmu_dx.transpose() << '\n';

    /* -------------------- Add New Data Test -------------------- */
    auto result_v_before_add_data = spgp_v.predict_cholesky(x_test2, true, true);
    std::cout << "[VariationalGPR Test] before add new data mu with " << spgp_v.m_N << " data:\n" << result_v2.y_mean << '\n';
    spgp_v.add_new_data(x_test2, result_v2.y_mean);
    auto result_v_add_data = spgp_v.predict_cholesky(x_test2, true, true);
    std::cout << "[VariationalGPR Test] after add new data mu with " << spgp_v.m_N << " data:\n" << result_v_add_data.y_mean << '\n';
}

void test_for_py_simulator(int sparse_method, bool normalize_gpr)
{
    std::cout << "[PySimulator Test] test with sparse_method " << sparse_method << ", and normalize is " << normalize_gpr << '\n';
    std::string file_path = "C:/Users/pc/Desktop/Personal/Code/GPRcpp/Log/py.txt";
    GPData data = read_sparse_gp_data_from_file(file_path, 12, 2, 260, 40, true);

    // Read for state v
    Eigen::RowVectorXd ard_length_scale_v = Eigen::RowVectorXd(12);
    ard_length_scale_v << 0.8658335733313588, 13.322692325094174, 6.407021934587417, 14.430933792473933, 0.8262508642621557, 12.956010458686743, 14.755220096026246, 17.347816965479478, 2.1033703167736517, 13.100624005598226, 14.435174572545298, 17.816198438786444;
    std::shared_ptr<kernel_base> constant_kernel_ptr_v = std::make_shared<constant_kernel>(15.938946986819923);
    std::shared_ptr<kernel_base> rbf_kernel_ptr_v = std::make_shared<rbf_kernel>(ard_length_scale_v);
	std::shared_ptr<kernel_base> my_kernel_v = std::make_shared<product_kernel>(constant_kernel_ptr_v, rbf_kernel_ptr_v);

    SparseGPR spgp_v(my_kernel_v, normalize_gpr);
    spgp_v.likelihood_varience = 0.00026610989521133605;
    if (sparse_method != 0) spgp_v.inference_method = 1;
    spgp_v.fit(data.m_feature, data.m_output.col(0), data.m_inducing_points);
	
    // Read for state w
    Eigen::RowVectorXd ard_length_scale_w = Eigen::RowVectorXd(12);
    // ard_length_scale_w << 179.89783828567562, 5.3518386878180575, 278.8128680361318, 37.15662167141493, 171.78692980352864, 176.27457390169434, 272.41892243231155, 258.7789715555678, 184.1016212824687, 5.335272339533333, 287.0791355720346, 220.52174051242275;
    ard_length_scale_w << 0.8658335733313588, 13.322692325094174, 6.407021934587417, 14.430933792473933, 0.8262508642621557, 12.956010458686743, 14.755220096026246, 17.347816965479478, 2.1033703167736517, 13.100624005598226, 14.435174572545298, 17.816198438786444;
    std::shared_ptr<kernel_base> constant_kernel_ptr_w = std::make_shared<constant_kernel>(15.938946986819923);//278.27107082447924
    std::shared_ptr<kernel_base> rbf_kernel_ptr_w = std::make_shared<rbf_kernel>(ard_length_scale_w);
	std::shared_ptr<kernel_base> my_kernel_w = std::make_shared<product_kernel>(constant_kernel_ptr_w, rbf_kernel_ptr_w);

    SparseGPR spgp_w(my_kernel_w, normalize_gpr);
    spgp_w.likelihood_varience = 0.00026610989521133605;//3.5873462385031854e-05
    if (sparse_method != 0) spgp_w.inference_method = 1;
    spgp_w.fit(data.m_feature, data.m_output.col(1), data.m_inducing_points_additional);

    // Check mean and var
    Eigen::MatrixXd x_test = spgp_v.m_X_train.block(0, 0, 4, spgp_v.m_X_train.cols());
    std::cout << "----------------------------\n";
    auto result_v = spgp_v.predict(x_test, true, false);
    auto result_w = spgp_w.predict(x_test, true, false);
    std::cout << "Mean result_v:\n" << result_v.y_mean << '\n';
    std::cout << "Cov result_v:\n" << result_v.y_cov << '\n';
    std::cout << "Mean result_w:\n" << result_w.y_mean << '\n';
    std::cout << "Cov result_w:\n" << result_w.y_cov << '\n';
    // std::cout << "dmu_dv:\n" << result_v.dmu_dx.transpose() << '\n';
    // std::cout << "dmu_dw:\n" << result_w.dmu_dx.transpose() << '\n';
    Eigen::MatrixXd x_test2 = Eigen::MatrixXd::Zero(1, 12);
    x_test2 << 0.00980392,0.0155402,0.192157,   0.304279,     0.0025, 0.00396425,        0.1,    0.15854,          0,          0,          0,          0;
    auto result_v2 = spgp_v.predict(x_test2, true, true);
    auto result_w2 = spgp_w.predict(x_test2, true, true);
    std::cout << "gradient result_v 2:\n" << result_v2.dmu_dx.transpose() << '\n';
    std::cout << "gradient result_w 2:\n" << result_w2.dmu_dx.transpose() << '\n';
}

/*
Test data for the sparse var DTC method is:
0.1 0.2 0.3 0.4 -0.5 1
0.5 0.6 0.7 0.8 0.5 2
0.9 1.0 1.1 1.2 1.5 3
0.1 0.2 0.3 0.4
0.5 0.6 0.7 0.8
*/
void update_cov(Eigen::MatrixXd & cov_before, double new_cov, bool update_covariance, const Eigen::MatrixXd & covariance)
{
    if (update_covariance)
    {
        const Eigen::MatrixXd cov_block = cov_before.block(0, 0, cov_before.rows() - 1, cov_before.cols() - 1);
        cov_before.block(1, 1, cov_before.rows() - 1, cov_before.cols() - 1).noalias() = cov_block;
        cov_before(0, 0) = new_cov;

        for (int i = 1; i < cov_before.rows(); i++)
        {
            cov_before(0, i) = covariance(i-1);
            cov_before(i, 0) = covariance(i-1);
        }
    }
    else
    {
        for (int i = cov_before.rows() - 1; i > 0; --i) {
            cov_before(i, i) = cov_before(i - 1, i - 1);
        }
        cov_before(0, 0) = new_cov;
    }
}

void test_uncertainty_propagation_20_times()
{
    std::string file_path = "C:/Users/pc/Desktop/Personal/Code/GPRcpp/Log/gazebo1.txt";
    GPData data = read_sparse_gp_data_from_file(file_path, 12, 2, 291, 40, true);
    Eigen::RowVectorXd ard_length_scale_v = Eigen::RowVectorXd(12);
    ard_length_scale_v << 100.017, 44.7549, 14.9609, 54.1276, 0.080009, 48.3174, 100.014, 100.496, 45.3872, 68.301, 16.327, 57.2148;
    std::shared_ptr<kernel_base> constant_kernel_ptr_v = std::make_shared<constant_kernel>(1.41831);
    std::shared_ptr<kernel_base> constant_kernel_ptr_2 = std::make_shared<constant_kernel>(1.21955e-11);
    std::shared_ptr<kernel_base> rbf_kernel_ptr_v = std::make_shared<rbf_kernel>(ard_length_scale_v);
    std::shared_ptr<kernel_base> rbf_kernel_ptr_2 = std::make_shared<rbf_kernel>(0.0101869);
    std::shared_ptr<kernel_base> realkernelPtr_1 = std::make_shared<product_kernel>(constant_kernel_ptr_v, rbf_kernel_ptr_v);
    std::shared_ptr<kernel_base> realkernelPtr_2 = std::make_shared<product_kernel>(constant_kernel_ptr_2, rbf_kernel_ptr_2);
    std::shared_ptr<kernel_base> realkernelPtr = std::make_shared<sum_kernel>(realkernelPtr_1, realkernelPtr_2);

    bool normalize_gpr = true;
    SparseGPR spgp_v(realkernelPtr, normalize_gpr);
    spgp_v.likelihood_varience = 0.33669;
    spgp_v.inference_method = 0;
    spgp_v.fit(data.m_feature, data.m_output.col(1), data.m_inducing_points_additional);

    // ExactGPR spgp_v(realkernelPtr, normalize_gpr);
    // spgp_v.fit(data.m_feature, data.m_output.col(1));

    Eigen::MatrixXd cov_before = Eigen::MatrixXd::Zero(data.m_feature.cols(), data.m_feature.cols());
    bool update_covariance = true;

    GPRcpp::gpr_results result;
}

void test_uncertainty_propagation()
{
    
}

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
    std::string file_path = "C:/Users/pc/Desktop/Personal/Code/GPRcpp/Log/test_util.txt";
    GPData data = read_sparse_gp_data_from_file(file_path, 4, 2, 3, 2, true);
    std::cout << data.m_feature << '\n';
    std::cout << data.m_output << '\n';
    std::cout << data.m_inducing_points << '\n';
    std::cout << data.m_inducing_points_additional << '\n';
    Eigen::RowVectorXd ard_length_scale_v = Eigen::RowVectorXd(4);
    ard_length_scale_v << 1, 2, 3, 4;
    double ard_length_scale_2 = 0.5;
    std::shared_ptr<kernel_base> constant_kernel_ptr = std::make_shared<constant_kernel>(0.5);
    std::shared_ptr<kernel_base> constant_kernel_ptr_2 = std::make_shared<constant_kernel>(2);
    std::shared_ptr<kernel_base> rbf_kernel_ptr = std::make_shared<rbf_kernel>(ard_length_scale_v);
    std::shared_ptr<kernel_base> rbf_kernel_ptr_2 = std::make_shared<rbf_kernel>(ard_length_scale_2);
    std::shared_ptr<kernel_base> realkernelPtr_1 = std::make_shared<product_kernel>(constant_kernel_ptr, rbf_kernel_ptr);
    std::shared_ptr<kernel_base> realkernelPtr_2 = std::make_shared<product_kernel>(constant_kernel_ptr_2, rbf_kernel_ptr_2);
    std::shared_ptr<kernel_base> realkernelPtr = std::make_shared<sum_kernel>(realkernelPtr_1, realkernelPtr_2);

    SparseGPR spgp_v(realkernelPtr, false);
    spgp_v.inference_method = 0;
    spgp_v.fit(data.m_feature, data.m_output.col(0), data.m_inducing_points);
    auto result = spgp_v.predict(data.m_feature, true);

    std::cout << "\n[varDTC] mean:\n" << result.y_mean << '\n';
    std::cout << "\n[varDTC] cov:\n" << result.y_cov << '\n';

    SparseGPR spgp_fict(realkernelPtr, false);
    spgp_fict.inference_method = 1;
    spgp_fict.fit(data.m_feature, data.m_output.col(0), data.m_inducing_points);
    auto result_fitc = spgp_fict.predict(data.m_feature, true);

    std::cout << "\n[FITC] mean:\n" << result_fitc.y_mean << '\n';
    std::cout << "\n[FITC] cov:\n" << result_fitc.y_cov << '\n';
}

void test_with_big_data(int iteration)
{
    std::string file_path = "C:/Users/pc/Desktop/Personal/Code/GPRcpp/Log/gazebo1.txt";
    GPData data = read_sparse_gp_data_from_file(file_path, 12, 2, 291, 40, true);
    Eigen::RowVectorXd ard_length_scale_v = Eigen::RowVectorXd(12);
    ard_length_scale_v << 100.017, 44.7549, 14.9609, 54.1276, 0.080009, 48.3174, 100.014, 100.496, 45.3872, 68.301, 16.327, 57.2148;
    std::shared_ptr<kernel_base> constant_kernel_ptr_v = std::make_shared<constant_kernel>(1.41831);
    std::shared_ptr<kernel_base> constant_kernel_ptr_2 = std::make_shared<constant_kernel>(1.21955e-11);
    std::shared_ptr<kernel_base> rbf_kernel_ptr_v = std::make_shared<rbf_kernel>(ard_length_scale_v);
    std::shared_ptr<kernel_base> rbf_kernel_ptr_2 = std::make_shared<rbf_kernel>(0.0101869);
    std::shared_ptr<kernel_base> realkernelPtr_1 = std::make_shared<product_kernel>(constant_kernel_ptr_v, rbf_kernel_ptr_v);
    std::shared_ptr<kernel_base> realkernelPtr_2 = std::make_shared<product_kernel>(constant_kernel_ptr_2, rbf_kernel_ptr_2);
    std::shared_ptr<kernel_base> realkernelPtr = std::make_shared<sum_kernel>(realkernelPtr_1, realkernelPtr_2);
    SparseGPR spgp_v(realkernelPtr, true);
    spgp_v.likelihood_varience = 0.33669;
    spgp_v.inference_method = 0;
    spgp_v.fit(data.m_feature, data.m_output.col(1), data.m_inducing_points_additional);
    auto result = spgp_v.predict(data.m_feature, true);

    auto params = spgp_v.kernel_->get_params();
    std::cout << "gpr_w params are: ";
    for (auto p: params) std::cout << p << ", ";
    std::cout << spgp_v.likelihood_varience << '\n';

    std::cout << "\nsparse mean:\n" << result.y_mean.block(0, 0, 4, 1) << '\n';
    std::cout << "\nsparse cov:\n" << result.y_cov.block(0, 0, 4, 4) << '\n';

    // ----------- A big data test ----------- //
    std::cout << "\n----------- A big data test -----------\n";
    ExactGPR egp(realkernelPtr, true);
    egp.fit(data.m_feature, data.m_output.col(1));
    auto result2 = egp.predict(data.m_feature, true);
    // egp.fit(data.m_feature.block(0, 0, 120, 12), data.m_output.block(0, 1, 120, 1));
    // auto result2 = egp.predict(data.m_feature.block(120, 0, 120, 12), true);
    std::cout << "\nexact mean:\n" << result2.y_mean.block(0, 0, 4, 1) << '\n';
    std::cout << "\nexact cov:\n" << result2.y_cov.block(0, 0, 4, 4) << '\n';

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iteration; i++)
    {
        const int index = std::min(decltype(data.m_feature.rows())(i), data.m_feature.rows() - 1);
        result2 = egp.predict(data.m_feature.row(index), true);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    std::cout << "\nsize of matrix element: " << sizeof(result2.y_cov(0, 0)) << " us" << '\n';
    std::cout << "\ncost time: " << duration << " us" << '\n';
}
