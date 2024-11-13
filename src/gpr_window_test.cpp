#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <vector>
#include "gprs/exact_gpr.h"
#include <chrono>

using namespace GPRcpp;

void read_input(std::vector<Eigen::RowVector4d> &v, int sample_size,
                int feature_window, const char* filename);

void get_features(const std::vector<Eigen::RowVector4d> &v, Eigen::MatrixXd &input_samples, 
                  Eigen::MatrixXd &output_samples_w,  Eigen::MatrixXd &output_samples_v,
                  int sample_size, int feature_window);

int main(int argc, char *argv[]) 
{
    ExactGPR gpr_obj_w;
    ExactGPR gpr_obj_v;
    int input_size_ = 1230;
    int train_sample_size = atoi(argv[1]);
    int test_sample_size = atoi(argv[2]);
    int input_window = atoi(argv[3]);
    int feature_window = atoi(argv[4]);
    int per_feature_size = 4;
    Eigen::RowVectorXd gpr_input;
    Eigen::RowVectorXd gpr_input_array;
    Eigen::RowVector4d gpr_feature_4d;

    Eigen::RowVectorXd ard_length_scale_ = Eigen::RowVectorXd::Ones(per_feature_size * feature_window);
    std::shared_ptr<kernel_base> constant_kernel_Ptr_w = std::make_shared<constant_kernel>(0.04688732092);
    std::shared_ptr<kernel_base> white_kernel_Ptr_w = std::make_shared<white_kernel>(6.28907471e-06);
    std::shared_ptr<kernel_base> rbf_kernel_Ptr_w = std::make_shared<rbf_kernel>(ard_length_scale_);
    std::shared_ptr<kernel_base> product_kernel_Ptr_w = std::make_shared<product_kernel>(constant_kernel_Ptr_w, rbf_kernel_Ptr_w);
    std::shared_ptr<kernel_base> kernelPtr_w = std::make_shared<sum_kernel>(product_kernel_Ptr_w, white_kernel_Ptr_w);
    std::shared_ptr<kernel_base> constant_kernel_Ptr_v = std::make_shared<constant_kernel>(0.16948565265283);
    std::shared_ptr<kernel_base> white_kernel_Ptr_v = std::make_shared<white_kernel>(1.088e-05);
    std::shared_ptr<kernel_base> rbf_kernel_Ptr_v = std::make_shared<rbf_kernel>(ard_length_scale_);
    std::shared_ptr<kernel_base> product_kernel_Ptr_v = std::make_shared<product_kernel>(constant_kernel_Ptr_v, rbf_kernel_Ptr_v);
    std::shared_ptr<kernel_base> kernelPtr_v = std::make_shared<sum_kernel>(product_kernel_Ptr_v, white_kernel_Ptr_v);

    gpr_obj_w = ExactGPR(kernelPtr_w);
    gpr_obj_v = ExactGPR(kernelPtr_v);

    {
        /***  Loading Data ***/
        gpr_input.resize(feature_window * per_feature_size);
        gpr_input.setZero();
        gpr_input_array.resize(input_window * per_feature_size);
        gpr_input_array.setZero();
        // Eigen::setNbThreads(1);
        std::vector<Eigen::RowVector4d> v;
        read_input(v, input_size_, feature_window, "../Log/jizhi.txt");
        std::cout << "Size of input: " << v.size() << std::endl;

        /***  Fitting Process ***/
        auto start = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXd input_samples;
        Eigen::MatrixXd output_samples_w;
        Eigen::MatrixXd output_samples_v;
        get_features(v, input_samples, output_samples_w, output_samples_v, train_sample_size, feature_window);
        gpr_obj_w.fit(input_samples, output_samples_w);
        gpr_obj_v.fit(input_samples, output_samples_v);     
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Fitting finished in :" << duration.count() << std::endl; 

        // double test_ptr = std::dynamic_pointer_cast<white_kernel>(white_kernel_Ptr)->test(); // 但是最初创建的时候就必须是指向这个类型的指针，否则会报错

        /***  Predicting Process ***/
        start = std::chrono::high_resolution_clock::now();
        gpr_results predict_w;
        gpr_results predict_v;
        Eigen::VectorXd y_mean_list_w(test_sample_size);
        Eigen::VectorXd y_mean_list_v(test_sample_size);
        get_features(v, input_samples, output_samples_w, output_samples_v, test_sample_size, feature_window);

        
        for (int i = 0; i < test_sample_size; ++i)
        {    
            predict_w = gpr_obj_w.predict(input_samples.row(i), true);
            predict_v = gpr_obj_v.predict(input_samples.row(i), true);
            y_mean_list_w(i) =  predict_w.y_mean(0);
            y_mean_list_v(i) =  predict_v.y_mean(0);

            if (test_sample_size < 5)
            {
                std::cout << "Input is: \n" << input_samples.row(i) << std::endl;
                std::cout << "Output w is: \n" << output_samples_w.row(i) << ", " << y_mean_list_w(i) << std::endl;
                std::cout << "Output v is: \n" << output_samples_v.row(i) << ", " << y_mean_list_v(i) << std::endl;
            }
        }
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Predict finished in :" << duration.count() << std::endl;
        std::cout << "Error sum w: " << (y_mean_list_w - output_samples_w).cwiseAbs().mean() << std::endl;
        std::cout << "Error sum v: " << (y_mean_list_v - output_samples_v).cwiseAbs().mean() << std::endl;

        std::cout << "T2 Error sum w: " << (y_mean_list_w - output_samples_w).segment(0, test_sample_size - input_window + 1).cwiseAbs().mean() << std::endl;
        std::cout << "T2 Error sum v: " << (y_mean_list_v - output_samples_v).segment(0, test_sample_size - input_window + 1).cwiseAbs().mean() << std::endl;

        if (test_sample_size > 30)
        {
            auto log_file = fopen("../Log/gpr.txt", "w");
            for (int i = 0; i < test_sample_size; ++i)
            {
                fprintf(log_file, "%f %f %f %f\n", y_mean_list_w(i), y_mean_list_v(i), output_samples_w(i), output_samples_v(i));
            }
            fclose(log_file);
            log_file = nullptr;
        }

        y_mean_list_w.setZero();
        y_mean_list_v.setZero();
        // auto log_file = fopen("../Log/jiazhi_lqr_1.txt", "w");
        for (int i = 0; i < test_sample_size; ++i)
        {    
            if (input_window > 1)
            {
                gpr_input_array.segment(0, (input_window - 1) * per_feature_size) = gpr_input_array.segment(per_feature_size, (input_window - 1) * per_feature_size);
            }
            gpr_feature_4d << input_samples.row(i)(0), input_samples.row(i)(1), input_samples.row(i)(2), input_samples.row(i)(3);
            gpr_input_array.segment((input_window - 1) * per_feature_size, per_feature_size) = gpr_feature_4d;
            gpr_input = gpr_input_array.segment(0, feature_window * per_feature_size);

            double feature_size = feature_window * per_feature_size;
            auto gpr_predict_result_w = gpr_obj_w.predict(gpr_input_array.segment(0, feature_size), true);
            auto gpr_predict_result_v = gpr_obj_v.predict(gpr_input_array.segment(0, feature_size), true);
            auto p_w_normal = gpr_predict_result_w.y_mean(0);
            auto p_v_normal = gpr_predict_result_v.y_mean(0);

            if (i >= input_window - 1)
            {
                y_mean_list_w(i - input_window + 1) = p_w_normal;
                y_mean_list_v(i - input_window + 1) = p_v_normal;
                // fprintf(log_file, "%f %f %f %f\n", p_w_normal, p_v_normal, output_samples_w(i - input_window + 1), output_samples_v(i - input_window + 1));
            }

            if (test_sample_size < 5)
            {
                std::cout << "Input is: \n" << gpr_input << std::endl;
                std::cout << "Input array is: \n" << gpr_input_array << std::endl;
                std::cout << "Output w is: \n" << output_samples_w(i) << ", " << p_w_normal << std::endl;
                std::cout << "Output v is: \n" << output_samples_v(i) << ", " << p_v_normal << std::endl;
            }
        }
        std::cout << "2 Error sum w: " << (y_mean_list_w - output_samples_w).segment(0, test_sample_size - input_window + 1).cwiseAbs().mean() << std::endl;
        std::cout << "2 Error sum v: " << (y_mean_list_v - output_samples_v).segment(0, test_sample_size - input_window + 1).cwiseAbs().mean() << std::endl;

        // fclose(log_file);
        // log_file = nullptr;
	gpr_feature_4d << 0.0100006, 0.00126642, 0.00989515, 0.00288956;
	std::cout << "Is there bug? " << gpr_feature_4d << std::endl;
	auto bug = gpr_obj_v.predict(gpr_feature_4d, true).y_mean(0);
	std::cout << bug << std::endl;

    }

    return 0;
}

void read_input(std::vector<Eigen::RowVector4d> &v, int sample_size,
                int feature_window, const char* filename) {
    std::ifstream inputFile(filename); // 打开文件
    Eigen::RowVector4d v_member;

    if (!inputFile) {
        std::cout << "无法打开文件!" << std::endl;
    }
    float num;
    int count = 0;
    int sample_count = 0;
    while (inputFile >> num) { // 一直读取直到文件结束
        if (count == 4) // per feature size
        {
        count = 0;
        v.push_back(v_member);
        sample_count++;
        }

        if (sample_count == sample_size + feature_window) break;
        v_member[count] = num;
        count++;
    }

    inputFile.close();
}

void get_features(const std::vector<Eigen::RowVector4d> &v, Eigen::MatrixXd &input_samples, 
                  Eigen::MatrixXd &output_samples_w,  Eigen::MatrixXd &output_samples_v,
                  int sample_size, int feature_window)
{
    input_samples.resize(sample_size, feature_window * 4);
    output_samples_w.resize(sample_size, 1);
    output_samples_v.resize(sample_size, 1);
    for (int i = feature_window; i < sample_size + feature_window; ++i)
    {
        for (int j = 0; j < feature_window; j++)
        {
            input_samples.row(i - feature_window).segment(j * 4, 4) = v[i - feature_window + j];
        }
        output_samples_w(i - feature_window) = v[i][3];
        output_samples_v(i - feature_window) = v[i][2];
    }
}
