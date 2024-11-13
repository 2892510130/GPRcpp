#include <Eigen/Core>

namespace GPRcpp
{

struct GPData
{
    Eigen::MatrixXd m_feature;
    Eigen::MatrixXd m_output;
    Eigen::MatrixXd m_inducing_points;
    Eigen::MatrixXd m_inducing_points_additional; // For now we just use this form, instead of a vector of inducing points
};


/**
 * @brief Read traning data from file to X_train_ and y_train_
 * @param path file path
 * @param feature_size the dimension of the features 
 * @param output_size the dimension of the outputs 
 * @param data_number the number of the traning data 
 * @return void
 * @note nothing
 */
GPData read_train_data_from_file(const std::string & path, size_t feature_size, size_t output_size, size_t data_number);

/**
 * @brief Read traning data from file to X_train_ and y_train_
 * @param path file path
 * @param feature_size the dimension of the features 
 * @param output_size the dimension of the outputs 
 * @param data_number the number of the traning data 
 * @param inducing_number the number of the inducing data 
 * @return void
 * @note nothing
 */
GPData read_sparse_gp_data_from_file(const std::string & path, size_t feature_size, size_t output_size, size_t data_number, size_t inducing_number, bool multiple_inducing_feature);

GPData read_sparse_gp_data_from_file(const std::string & path, size_t feature_size, size_t output_size, size_t data_number, size_t inducing_number);

} // namespace GPRcpp