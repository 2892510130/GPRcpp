#include "utils/file_operation.h"
#include <fstream>

namespace GPRcpp
{

GPData read_train_data_from_file(const std::string & path, size_t feature_size, size_t output_size, size_t data_number)
{
    std::ifstream data_file(path);
    if (!data_file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    GPData gp_data;
    gp_data.m_feature = Eigen::MatrixXd::Zero(data_number, feature_size);
    gp_data.m_output = Eigen::MatrixXd::Zero(data_number, output_size);

    std::string line;
    size_t row = 0;
    while (std::getline(data_file, line) && row < data_number) {
        std::stringstream ss(line);
        for (size_t col = 0; col < feature_size; ++col) {
            ss >> gp_data.m_feature(row, col);
        }
        for (size_t col = 0; col < output_size; ++col) {
            ss >> gp_data.m_output(row, col);
        }
        ++row;
    }

    data_file.close();
    return gp_data;
}

GPData read_sparse_gp_data_from_file(const std::string & path, size_t feature_size, size_t output_size, size_t data_number, size_t inducing_number, bool multiple_inducing_feature)
{
    std::ifstream data_file(path);
    if (!data_file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    GPData gp_data;
    gp_data.m_feature = Eigen::MatrixXd::Zero(data_number, feature_size);
    gp_data.m_output = Eigen::MatrixXd::Zero(data_number, output_size);
    gp_data.m_inducing_points = Eigen::MatrixXd::Zero(inducing_number, feature_size);

    std::string line;
    size_t row = 0;
    size_t inducing_feature_number;
    if (multiple_inducing_feature) 
    {
        inducing_feature_number = inducing_number * 2;
        gp_data.m_inducing_points_additional = Eigen::MatrixXd::Zero(inducing_number, feature_size);
    }
    else inducing_feature_number = inducing_number;

    while (std::getline(data_file, line) && row < data_number + inducing_feature_number) {
        std::stringstream ss(line);
        if (row < data_number)
        {
            for (size_t col = 0; col < feature_size; ++col) {
                ss >> gp_data.m_feature(row, col);
            }
            for (size_t col = 0; col < output_size; ++col) {
                ss >> gp_data.m_output(row, col);
            }
        }
        else if (row < data_number + inducing_number)
        {
            for (size_t col = 0; col < feature_size; ++col) {
                ss >> gp_data.m_inducing_points(row-data_number, col);
            }
        }
        else
        {
            for (size_t col = 0; col < feature_size; ++col) {
                ss >> gp_data.m_inducing_points_additional(row-data_number-inducing_number, col);
            }
        }
        ++row;
    }

    data_file.close();
    return gp_data;
}

GPData read_sparse_gp_data_from_file(const std::string & path, size_t feature_size, size_t output_size, size_t data_number, size_t inducing_number)
{
    return read_sparse_gp_data_from_file(path, feature_size, output_size, data_number, inducing_number, false);
}

} // namespace GPRcpp