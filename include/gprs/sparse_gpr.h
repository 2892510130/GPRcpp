#pragma once
#include "gprs/gpr.h"

namespace GPRcpp
{

class SparseGPR: public gpr
{
public:
    using gpr::gpr;
    ~SparseGPR();
    void fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train, const Eigen::MatrixXd & inducing_points) override;
    void fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train) override;
    
    /*
     * 
     *
     *
     * */
    void fit_with_dtc();
    
    void fit_with_fitc();
    gpr_results predict(const Eigen::MatrixXd & X_test) override;
    gpr_results predict(const Eigen::MatrixXd & X_test, bool return_cov, bool compute_jac = false) override;

    void save_data(const std::string& filename) override;
    void load_data(const std::string& filename) override;
    void save_matrix_to_file(std::ofstream &file, const Eigen::MatrixXd & matrix);
    void load_matrix_from_file(std::ifstream &file, Eigen::MatrixXd & matrix);
    void load_vector_from_file(std::ifstream &file, Eigen::RowVectorXd & vector);

public:
    Eigen::MatrixXd m_inducing_point;
    Eigen::MatrixXd woodbury_inv;
    int sparse_method = 0; // 0 for varDTC, 1 for FITC
};

} // namespace GPRcpp
