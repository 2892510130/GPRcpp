#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "gprs/sparse_gpr.h"
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

void test_for_py_simulator();

int main(int argc, char *argv[]) 
{
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
        test_for_py_simulator();
    }

    return 0;
}

void test_for_py_simulator()
{
    std::string file_path = "C:/Users/pc/Desktop/Personal/Code/GPRcpp/Log/py.txt";
    GPData data = read_sparse_gp_data_from_file(file_path, 12, 2, 129, 40, true);
    bool normalize_gpr = true;

    // Read for state v
    Eigen::RowVectorXd ard_length_scale_v = Eigen::RowVectorXd(12);
    ard_length_scale_v << 0.14262398286403705, 15.347526358466826, 0.965777312794053, 17.382834917506045, 3.0938293556819376, 7.52548756901615, 7.2464594032077665, 13.605523567836714, 3.174681803889279, 4.728858179375551, 0.9709302889247808, 12.781116298767273;
    std::shared_ptr<kernel_base> constant_kernel_ptr_v = std::make_shared<constant_kernel>(6.994427226495026);
    std::shared_ptr<kernel_base> rbf_kernel_ptr_v = std::make_shared<rbf_kernel>(ard_length_scale_v);
	std::shared_ptr<kernel_base> my_kernel_v = std::make_shared<product_kernel>(constant_kernel_ptr_v, rbf_kernel_ptr_v);

    SparseGPR spgp_v(my_kernel_v, normalize_gpr);
    spgp_v.likelihood_varience = 0.10276029249210838;
    spgp_v.fit(data.m_feature, data.m_output.col(0), data.m_inducing_points);
	
    // Read for state w
    Eigen::RowVectorXd ard_length_scale_w = Eigen::RowVectorXd(12);
    ard_length_scale_w << 359.0152783762107, 7.645096386212836, 361.3205472864519, 51.631464829145116, 363.64511914908553, 397.7885897266975, 475.045676162974, 633.1474014306848, 318.0744148727092, 7.555457111478175, 580.4603250244658, 261.4863448698691;
    std::shared_ptr<kernel_base> constant_kernel_ptr_w = std::make_shared<constant_kernel>(851.377738351321);
    std::shared_ptr<kernel_base> rbf_kernel_ptr_w = std::make_shared<rbf_kernel>(ard_length_scale_w);
	std::shared_ptr<kernel_base> my_kernel_w = std::make_shared<product_kernel>(constant_kernel_ptr_w, rbf_kernel_ptr_w);

    SparseGPR spgp_w(my_kernel_w, normalize_gpr);
    spgp_w.likelihood_varience = 0.004372263083734497;
    spgp_w.fit(data.m_feature, data.m_output.col(1), data.m_inducing_points_additional);

    // Check mean and var
    Eigen::MatrixXd x_test = Eigen::MatrixXd::Zero(1, 12);
    auto result_v = spgp_v.predict(x_test, true, true);
    auto result_w = spgp_w.predict(x_test, true, true);
    std::cout << "Mean result_v:\n" << result_v.y_mean << '\n';
    std::cout << "Cov result_v:\n" << result_v.y_cov << '\n';
    std::cout << "Mean result_w:\n" << result_w.y_mean << '\n';
    std::cout << "Cov result_w:\n" << result_w.y_cov << '\n';
    std::cout << "dmu_dv:\n" << result_v.dmu_dx.transpose() << '\n';
    std::cout << "dmu_dw:\n" << result_w.dmu_dx.transpose() << '\n';
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

    for (int i = 0; i < 10; i++)
    {
        result = spgp_v.predict_at_uncertain_input(data.m_feature.row(i), cov_before, update_covariance, false);
        double new_cov = result.y_cov(0, 0) + 1 * cov_before(0 , 0);
        update_cov(cov_before, new_cov, update_covariance, result.y_covariance);
        std::cout << "\n[ExactGPR] final sigma " << i << ":\n" << cov_before << '\n';
    }
}

void test_uncertainty_propagation()
{
    std::string file_path = "C:\\Users\\pc\\Desktop\\Personal\\Code\\GPRcpp\\Log\\test_util.txt";
    GPData data = read_sparse_gp_data_from_file(file_path, 4, 2, 3, 2, true);

    Eigen::RowVectorXd ard_length_scale_v = Eigen::RowVectorXd(4);
    ard_length_scale_v << 1, 2, 3, 4;
    double ard_length_scale_2 = 0.5;

    std::shared_ptr<kernel_base> rbf_kernel_ptr = std::make_shared<rbf_kernel>(ard_length_scale_v);
    std::shared_ptr<kernel_base> constant_kernel_ptr = std::make_shared<constant_kernel>(0.5);
    std::shared_ptr<kernel_base> white_kernel_ptr = std::make_shared<white_kernel>(0.33);
    std::shared_ptr<kernel_base> real_kernel_1 = std::make_shared<product_kernel>(constant_kernel_ptr, rbf_kernel_ptr);
    std::shared_ptr<kernel_base> real_kernel_2 = std::make_shared<sum_kernel>(real_kernel_1, white_kernel_ptr);

    // For 1 sample with dimension D, this dk_dx will return N * D matrix
    Eigen::MatrixXd dk_dx_rbf =  rbf_kernel_ptr->dk_dx(data.m_feature, data.m_feature.row(0));
    Eigen::MatrixXd dk_dx_const =  constant_kernel_ptr->dk_dx(data.m_feature, data.m_feature.row(0));
    Eigen::MatrixXd dk_dx_white =  white_kernel_ptr->dk_dx(data.m_feature, data.m_feature.row(0));
    Eigen::MatrixXd dk_dx_product =  real_kernel_1->dk_dx(data.m_feature, data.m_feature.row(0));
    Eigen::MatrixXd dk_dx_sum =  real_kernel_2->dk_dx(data.m_feature, data.m_feature.row(0));

    std::cout << "rbf dk_dx is:\n" << dk_dx_rbf << '\n';
    std::cout << "const dk_dx is:\n" << dk_dx_const << '\n';
    std::cout << "white dk_dx is:\n" << dk_dx_white << '\n';
    std::cout << "product dk_dx is:\n" << dk_dx_product << '\n';
    std::cout << "sum dk_dx is:\n" << dk_dx_sum << '\n';

    ExactGPR gpr(real_kernel_2, false);

    gpr.fit(data.m_feature, data.m_output.col(0));

    Eigen::MatrixXd cov_before = Eigen::MatrixXd::Zero(data.m_feature.cols(), data.m_feature.cols());
    bool update_covariance = true;

    auto result_1 = gpr.predict_at_uncertain_input(data.m_feature.row(0), cov_before, update_covariance, false);
    update_cov(cov_before, result_1.y_cov(0, 0), update_covariance, result_1.y_covariance);
    std::cout << "[ExactGPR] final mean:\n" << result_1.y_mean << '\n';
    std::cout << "[ExactGPR] final cov:\n" << result_1.y_cov << '\n';
    std::cout << "[ExactGPR] final sigma 1:\n" << cov_before << '\n';
    
    result_1 = gpr.predict_at_uncertain_input(data.m_feature.row(1), cov_before, update_covariance, false);
    update_cov(cov_before, result_1.y_cov(0, 0), update_covariance, result_1.y_covariance);
    std::cout << "\n[ExactGPR] final sigma 2:\n" << cov_before << '\n';

    result_1 = gpr.predict_at_uncertain_input(data.m_feature.row(2), cov_before, update_covariance, false);
    update_cov(cov_before, result_1.y_cov(0, 0), update_covariance, result_1.y_covariance);
    std::cout << "\n[ExactGPR] final sigma 3:\n" << cov_before << '\n';

    result_1 = gpr.predict_at_uncertain_input(data.m_feature.row(1), cov_before, update_covariance, false);
    update_cov(cov_before, result_1.y_cov(0, 0), update_covariance, result_1.y_covariance);
    std::cout << "\n[ExactGPR] final sigma 4:\n" << cov_before << '\n';

    result_1 = gpr.predict_at_uncertain_input(data.m_feature.row(0), cov_before, update_covariance, false);
    update_cov(cov_before, result_1.y_cov(0, 0), update_covariance, result_1.y_covariance);
    std::cout << "\n[ExactGPR] final sigma 5:\n" << cov_before << '\n';

    std::shared_ptr<kernel_base> rbf_kernel_ptr_2 = std::make_shared<rbf_kernel>(2);
    std::shared_ptr<kernel_base> constant_kernel_ptr_2 = std::make_shared<constant_kernel>(0.5);
    std::shared_ptr<kernel_base> real_kernel_3 = std::make_shared<product_kernel>(constant_kernel_ptr_2, rbf_kernel_ptr_2);
    SparseGPR spgp_v(real_kernel_3, false);
    spgp_v.inference_method = 1;
    spgp_v.fit(data.m_feature, data.m_output.col(0), data.m_inducing_points);

    Eigen::MatrixXd cov_before_sparse = Eigen::MatrixXd::Zero(data.m_feature.cols(), data.m_feature.cols());
    result_1 = spgp_v.predict_at_uncertain_input(data.m_feature.row(0), cov_before_sparse, update_covariance, false);
    update_cov(cov_before_sparse, result_1.y_cov(0, 0), update_covariance, result_1.y_covariance);
    std::cout << "\n[SparseGPR] mean:\n" << result_1.y_mean << '\n';
    std::cout << "\n[SparseGPR] cov:\n" << result_1.y_cov << '\n';
    std::cout << "\n[SparseGPR] final sigma 1:\n" << cov_before_sparse << '\n';

    std::shared_ptr<kernel_base> real_kernel_4 = std::make_shared<sum_kernel>(real_kernel_1, real_kernel_3);
    std::cout << "\n[KernleTest] kernel_2:\n" << real_kernel_2->dk_dx(data.m_inducing_points, data.m_feature.row(0)) << '\n';
    std::cout << "\n[KernleTest] kernel_3:\n" << real_kernel_3->dk_dx(data.m_inducing_points, data.m_feature.row(0)) << '\n';
    std::cout << "\n[KernleTest] kernel_2+3:\n" << real_kernel_2->dk_dx(data.m_inducing_points, data.m_feature.row(0)) + real_kernel_3->dk_dx(data.m_inducing_points, data.m_feature.row(0)) << '\n';
    std::cout << "\n[KernleTest] kernel_4:\n" << real_kernel_4->dk_dx(data.m_inducing_points, data.m_feature.row(0)) << '\n';


    // gpr.predict_at_uncertain_input(data.m_feature, cov_before); // Will cause error
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
