#pragma once
#include "gprs/gpr.h"

namespace GPRcpp
{

class VariationalGPR: public gpr
{
public:
    using gpr::gpr;
    ~VariationalGPR();
    void fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train, const Eigen::MatrixXd & inducing_points) override;
    void fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train) override;
    void fit_with_dtc();
    void fit_with_fitc();
    gpr_results predict(const Eigen::MatrixXd & X_test) override;
    gpr_results predict(const Eigen::MatrixXd & X_test, bool return_cov, bool compute_jac = false) override;
    gpr_results predict_cholesky(const Eigen::MatrixXd & X_test, bool return_cov, bool compute_jac = false);

    void add_new_data(const Eigen::MatrixXd & X_new, const Eigen::MatrixXd & Y_new);
    void remove_data(int remove_number = 1);
    void add_new_inducing_data(const Eigen::MatrixXd & U_new);
    void add_new_inducing_quick(const Eigen::MatrixXd & U_new); // if we only use mu su to predict
    void update_L();
    void update_Luu(const Eigen::MatrixXd & Kun, const Eigen::MatrixXd & Knn);
    void add_mu_su(const Eigen::MatrixXd & Kun, const Eigen::MatrixXd & Knn);
    void update_mu_su(const Eigen::MatrixXd & X_new, const Eigen::MatrixXd & Y_new);

public:
    Eigen::MatrixXd m_inducing_point;

    size_t m_N, m_M, m_D;

    Eigen::MatrixXd m_Luu;
    Eigen::MatrixXd m_mu;
    Eigen::MatrixXd m_Su;

    bool m_L_updated = true;
    Eigen::MatrixXd m_Kuf; // needed for the add/remove function
    Eigen::MatrixXd m_W; // need for the cholesky predict method
    Eigen::MatrixXd m_diag; // needed for the add/remove function
    Eigen::MatrixXd m_diag_inv; // needed for the add/remove function
    Eigen::MatrixXd m_W_diag_inv_y; // need for the cholesky predict method
    Eigen::MatrixXd m_L; // need for the cholesky predict method
    int sparse_method = 0; // 0 for varDTC, 1 for FITC
};

} // namespace GPRcpp
