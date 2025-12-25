#include "gprs/variational_gpr.h"
#include <iostream>
#include <Eigen/Dense>

namespace GPRcpp
{
    VariationalGPR::~VariationalGPR()
    {

    }

    void VariationalGPR::fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train, const Eigen::MatrixXd & inducing_points)
    {
        std::cout << "[GPR INFO]: fitting variational GPR with method " << inference_method << '\n';
        has_x_train_ = true;
        m_N = X_train.rows();
        m_D = X_train.cols();
        m_M = inducing_points.rows();

        m_X_train = Eigen::MatrixXd::Zero(2 * m_N, m_D);
        m_y_train = Eigen::MatrixXd::Zero(2 * m_N, 1);
        m_inducing_point = Eigen::MatrixXd::Zero(2 * m_M, m_D);
        m_Kuf = Eigen::MatrixXd::Zero(2 * m_M, 2 * m_N);
        m_W = Eigen::MatrixXd::Zero(2 * m_M, 2 * m_N);
        m_Luu = Eigen::MatrixXd::Zero(2 * m_M, 2 * m_M);
        m_diag = Eigen::MatrixXd::Zero(1, 2 * m_N);
        m_diag_inv = Eigen::MatrixXd::Zero(1, 2 * m_N);
        m_mu = Eigen::MatrixXd::Zero(2 * m_N, 1);
        m_Su = Eigen::MatrixXd::Zero(2 * m_N, 2 * m_N);
        
        if (normalize_y_)
        {
            y_train_mean_ = y_train.colwise().mean();
            y_train_std_ = ((y_train.rowwise() - y_train_mean_).array().square().colwise().sum() / y_train.rows()).sqrt();
            m_y_train.block(0, 0, m_N, 1) = (y_train.rowwise() - y_train_mean_).array().rowwise() / y_train_std_.array();
        }
        else
        {
            m_y_train.block(0, 0, m_N, 1) = y_train;
        }

        m_X_train.block(0, 0, m_N, m_D) = X_train;
        m_inducing_point.block(0, 0, m_M, m_D) = inducing_points;

        if (inference_method == 0) fit_with_dtc();
        else if (inference_method == 1) fit_with_fitc();
    }

    void VariationalGPR::fit(const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & y_train)
    {
        std::cout << "<!!!! It is sparse GPR, please use fit function with inducing points !!!!>" << '\n';
        throw std::runtime_error("Not implemented");
    }

    void VariationalGPR::fit_with_fitc()
    {
        Eigen::MatrixXd Kuu = kernel_->evaluate(m_inducing_point.block(0, 0, m_M, m_D));
        Kuu = (Kuu + Kuu.transpose()) / 2.0;
        m_Kuf.block(0, 0, m_M, m_N) = kernel_->evaluate(m_inducing_point.block(0, 0, m_M, m_D), m_X_train.block(0, 0, m_N, m_D));

        Eigen::MatrixXd jitter = 1e-6 * Eigen::MatrixXd::Identity(m_M, m_M); // We hard code jitter to be 1e-6 here
        m_Luu.block(0, 0, m_M, m_M) = (Kuu + jitter).llt().matrixL();
        Eigen::MatrixXd Luu = m_Luu.block(0, 0, m_M, m_M);

        m_W.block(0, 0, m_M, m_N) = Luu.triangularView<Eigen::Lower>().solve(m_Kuf.block(0, 0, m_M, m_N));
        Eigen::MatrixXd W = m_W.block(0, 0, m_M, m_N);

        Eigen::MatrixXd Kff_diag = kernel_->k_diag(m_X_train.block(0, 0, m_N, m_D)).transpose();
        Eigen::MatrixXd Qff_diag = W.array().square().colwise().sum();
        m_diag.block(0, 0, 1, m_N) = (Kff_diag - Qff_diag).array() + likelihood_varience;
        m_diag_inv.block(0, 0, 1, m_N) = 1.0 / m_diag.block(0, 0, 1, m_N).array();
        Eigen::MatrixXd W_diag_inv = W.array() * m_diag_inv.block(0, 0, 1, m_N).array().replicate(m_M, 1);
        m_W_diag_inv_y = W_diag_inv * m_y_train.block(0, 0, m_N, 1);

        Eigen::MatrixXd K = W_diag_inv * W.transpose();
        K = (K + K.transpose()) / 2.0 + Eigen::MatrixXd::Identity(m_M, m_M);
        m_L = K.llt().matrixL();

        // Get the mu and Su
        Eigen::MatrixXd tmp_1 = m_L.triangularView<Eigen::Lower>().solve(Luu.transpose());
        m_Su.block(0, 0, m_M, m_M) = tmp_1.transpose() * tmp_1;
        Eigen::MatrixXd tmp_2 = (m_Kuf.block(0, 0, m_M, m_N).array() * m_diag_inv.block(0, 0, 1, m_N).array().replicate(m_M, 1)).matrix() * m_y_train.block(0, 0, m_N, 1);
        Eigen::MatrixXd rhs = Luu.triangularView<Eigen::Lower>().solve(tmp_2);
        m_mu.block(0, 0, m_M, 1) = Luu.triangularView<Eigen::Lower>().solve(m_Su.block(0, 0, m_M, m_M).transpose()).transpose() * rhs;
    }

    void VariationalGPR::fit_with_dtc()
    {
        Eigen::MatrixXd Kuu = kernel_->evaluate(m_inducing_point.block(0, 0, m_M, m_D));
        Kuu = (Kuu + Kuu.transpose()) / 2.0;
        m_Kuf.block(0, 0, m_M, m_N) = kernel_->evaluate(m_inducing_point.block(0, 0, m_M, m_D), m_X_train.block(0, 0, m_N, m_D));

        Eigen::MatrixXd jitter = alpha_ * Eigen::MatrixXd::Identity(m_M, m_M);
        m_Luu.block(0, 0, m_M, m_M) = (Kuu + jitter).llt().matrixL();
        Eigen::MatrixXd Luu = m_Luu.block(0, 0, m_M, m_M);

        m_W.block(0, 0, m_M, m_N) = Luu.triangularView<Eigen::Lower>().solve(m_Kuf.block(0, 0, m_M, m_N));
        Eigen::MatrixXd W = m_W.block(0, 0, m_M, m_N);

        Eigen::MatrixXd W_diag_inv = W.array() / likelihood_varience;
        m_W_diag_inv_y = W_diag_inv * m_y_train.block(0, 0, m_N, 1);

        Eigen::MatrixXd K = W_diag_inv * W.transpose();
        K = (K + K.transpose()) / 2.0 + Eigen::MatrixXd::Identity(m_M, m_M);
        m_L = K.llt().matrixL();

        // Get the mu and Su
        Eigen::MatrixXd tmp_1 = m_L.triangularView<Eigen::Lower>().solve(Luu.transpose());
        m_Su.block(0, 0, m_M, m_M) = tmp_1.transpose() * tmp_1;
        Eigen::MatrixXd tmp_2 = (m_Kuf.block(0, 0, m_M, m_N) / likelihood_varience) * m_y_train.block(0, 0, m_N, 1);
        Eigen::MatrixXd rhs = Luu.triangularView<Eigen::Lower>().solve(tmp_2);
        m_mu.block(0, 0, m_M, 1) = Luu.triangularView<Eigen::Lower>().solve(m_Su.block(0, 0, m_M, m_M).transpose()).transpose() * rhs;
    }

    gpr_results VariationalGPR::predict(const Eigen::MatrixXd & X_test)
    {
        return predict(X_test, false);
    }

    gpr_results VariationalGPR::predict(const Eigen::MatrixXd & X_test, bool return_cov, bool compute_jac)
    {
        Eigen::MatrixXd Luu = m_Luu.block(0, 0, m_M, m_M);
        Eigen::MatrixXd Kss = kernel_->evaluate(X_test);
        Eigen::MatrixXd Ksu = kernel_->evaluate(X_test, m_inducing_point.block(0, 0, m_M, m_D));
        Eigen::MatrixXd Ws = Luu.triangularView<Eigen::Lower>().solve(Ksu.transpose());
        Eigen::MatrixXd tmp = Luu.transpose().triangularView<Eigen::Upper>().solve(Ws);
        
        Eigen::MatrixXd rhs = Luu.triangularView<Eigen::Lower>().solve(m_mu.block(0, 0, m_M, 1));
        results_.y_mean = Ws.transpose() * rhs;

        if (return_cov)
            results_.y_cov = Kss - Ws.transpose() * Ws + tmp.transpose() * m_Su.block(0, 0, m_M, m_M) * tmp;

        if (compute_jac)
        {
            Eigen::MatrixXd dk_dx = kernel_->dk_dx(m_inducing_point.block(0, 0, m_M, m_D), X_test);
            Eigen::MatrixXd lhs = Luu.triangularView<Eigen::Lower>().solve(dk_dx);
            results_.dmu_dx = lhs.transpose() * rhs;
            if (normalize_y_)
            {
                results_.dmu_dx = (results_.dmu_dx.array().rowwise() * y_train_std_.array());
            }
        }

        if (normalize_y_)
        {
            results_.y_mean = (results_.y_mean.array().rowwise() * y_train_std_.array()).rowwise() + y_train_mean_.array();
            results_.y_cov = results_.y_cov * y_train_std_(0) * y_train_std_(0); // Only for 1D output.
        }
        
        return results_;
    }

    gpr_results VariationalGPR::predict_cholesky(const Eigen::MatrixXd & X_test, bool return_cov, bool compute_jac)
    {
        Eigen::MatrixXd Luu = m_Luu.block(0, 0, m_M, m_M);
        Eigen::MatrixXd Kss = kernel_->evaluate(X_test);
        Eigen::MatrixXd Ksu = kernel_->evaluate(X_test, m_inducing_point.block(0, 0, m_M, m_D));
        Eigen::MatrixXd Ws = Luu.triangularView<Eigen::Lower>().solve(Ksu.transpose());

        update_L();

        Eigen::MatrixXd L_inv_Ws = m_L.triangularView<Eigen::Lower>().solve(Ws);
        Eigen::MatrixXd L_inv_W_d_inv_y = m_L.triangularView<Eigen::Lower>().solve(m_W_diag_inv_y);

        results_.y_mean = L_inv_Ws.transpose() * L_inv_W_d_inv_y;
        if (return_cov) results_.y_cov = Kss - Ws.transpose() * Ws + L_inv_Ws.transpose() * L_inv_Ws;
        return results_;
    }

    void VariationalGPR::update_L()
    {
        if (m_L_updated == false)
        {
            Eigen::MatrixXd W_diag_inv;
            if (inference_method == 0)
            {
                W_diag_inv = m_W.block(0, 0, m_M, m_N) / likelihood_varience;
            }
            else
            {
                W_diag_inv = m_W.block(0, 0, m_M, m_N).array() * m_diag_inv.block(0, 0, 1, m_N).array().replicate(m_M, 1);
            }
            Eigen::MatrixXd K = W_diag_inv * m_W.block(0, 0, m_M, m_N).transpose();
            K = (K + K.transpose()) / 2.0 + Eigen::MatrixXd::Identity(m_M, m_M);
            m_L = K.llt().matrixL();
            m_L_updated = true;
        }
    }

    void VariationalGPR::add_new_data(const Eigen::MatrixXd & X_new, const Eigen::MatrixXd & Y_new)
    {
        size_t new_N = X_new.rows();
        m_X_train.block(m_N, 0, new_N, m_D) = X_new;
        m_y_train.block(m_N, 0, new_N, 1) = Y_new;

        Eigen::MatrixXd Kuf_new = kernel_->evaluate(m_inducing_point.block(0, 0, m_M, m_D), X_new);
        Eigen::MatrixXd W_new = m_Luu.block(0, 0, m_M, m_M).triangularView<Eigen::Lower>().solve(Kuf_new);
        m_Kuf.block(0, m_N, m_M, new_N) = Kuf_new;
        m_W.block(0, m_N, m_M, new_N) = W_new;

        Eigen::MatrixXd W_diag_inv_y_new;
        if (inference_method == 1)
        {
            Eigen::MatrixXd Kff_diag_new = kernel_->k_diag(X_new).transpose();
            Eigen::MatrixXd Qff_diag_new = W_new.array().square().colwise().sum();
            Eigen::MatrixXd diag_new = (Kff_diag_new - Qff_diag_new).array() + likelihood_varience;
            Eigen::MatrixXd diag_inv_new = 1.0 / diag_new.array();
            m_diag.block(0, m_N, 1, new_N) = diag_new;
            m_diag_inv.block(0, m_N, 1, new_N) = diag_inv_new;
            W_diag_inv_y_new = (W_new.array() * diag_inv_new.array().replicate(m_M, 1)).matrix() * Y_new;
        }
        else
        {
            W_diag_inv_y_new = (W_new / likelihood_varience) * Y_new;
        }

        m_W_diag_inv_y += W_diag_inv_y_new;

        m_L_updated = false;

        m_N += new_N;

        // For the test
        // std::cout << Kuf_new << '\n';
        // std::cout << W_new << '\n';
        // std::cout << diag_new << '\n';
        // std::cout << W_diag_inv_y_new << '\n';
    }

    void VariationalGPR::remove_data(int remove_number)
    {
        // If we choose not to copy the rest data to the front (which will take O(NM) time), 
        // we can just mark the removed data as invalid, and skip them during prediction.
        // Which will again take too much space.
    }

    void VariationalGPR::add_new_inducing_data(const Eigen::MatrixXd & U_new)
    {
        size_t new_M = U_new.rows();
        if (new_M > 1) std::runtime_error("Currently only support adding one inducing point at a time.");

        m_inducing_point.block(m_M, 0, new_M, m_D) = U_new;

        Eigen::MatrixXd Kun = kernel_->evaluate(m_inducing_point.block(0, 0, m_M, m_D), U_new);
        Eigen::MatrixXd Knn = kernel_->evaluate(U_new) + alpha_ * Eigen::MatrixXd::Identity(new_M, new_M);

        add_mu_su(Kun, Knn);

        update_Luu(Kun, Knn);

        Eigen::MatrixXd Knf = kernel_->evaluate(U_new, m_X_train.block(0, 0, m_N, m_D));
        m_Kuf.block(m_M, 0, new_M, m_N) = Knf;

        Eigen::MatrixXd W_new = (Knf - m_Luu.block(m_M, 0, new_M, m_M) * m_W.block(0, 0, m_M, m_N)).array() / m_Luu(m_M, m_M);
        m_W.block(m_M, 0, new_M, m_N) = W_new;

        if (inference_method == 1)
        {
            Eigen::MatrixXd Qff_diag_new = W_new.array().square().colwise().sum();
            m_diag.block(0, 0, 1, m_N) = m_diag.block(0, 0, 1, m_N).array() - Qff_diag_new.array();
            m_diag_inv.block(0, 0, 1, m_N) = 1.0 / m_diag.block(0, 0, 1, m_N).array();
        }

        m_M += new_M;

        Eigen::MatrixXd W_diag_inv;
        if (inference_method == 1)
        {
            W_diag_inv = m_W.block(0, 0, m_M, m_N).array() * m_diag_inv.block(0, 0, 1, m_N).array().replicate(m_M, 1);
        }
        else
        {
            W_diag_inv = m_W.block(0, 0, m_M, m_N) / likelihood_varience;
        }
        m_W_diag_inv_y = W_diag_inv * m_y_train.block(0, 0, m_N, 1);

        m_L_updated = false;
    }

    void VariationalGPR::add_new_inducing_quick(const Eigen::MatrixXd & U_new)
    {
        size_t new_M = U_new.rows();
        if (new_M > 1) std::runtime_error("Currently only support adding one inducing point at a time.");

        m_inducing_point.block(m_M, 0, new_M, m_D) = U_new;

        Eigen::MatrixXd Kun = kernel_->evaluate(m_inducing_point.block(0, 0, m_M, m_D), U_new);
        Eigen::MatrixXd Knn = kernel_->evaluate(U_new) + alpha_ * Eigen::MatrixXd::Identity(new_M, new_M);

        add_mu_su(Kun, Knn);

        update_Luu(Kun, Knn);

        m_M += new_M;
    }

    void VariationalGPR::update_Luu(const Eigen::MatrixXd & Kun, const Eigen::MatrixXd & Knn)
    {
        Eigen::MatrixXd Luu_old = m_Luu.block(0, 0, m_M, m_M);
        size_t new_M = Kun.cols();
        Eigen::MatrixXd A = Luu_old.triangularView<Eigen::Lower>().solve(Kun);
        Eigen::MatrixXd S = (Knn - A.transpose() * A).array().sqrt();
        m_Luu.block(m_M, 0, new_M, m_M) = A.transpose();
        m_Luu.block(m_M, m_M, new_M, new_M) = S;
    }

    void VariationalGPR::add_mu_su(const Eigen::MatrixXd & Kun, const Eigen::MatrixXd & Knn)
    {
        Eigen::MatrixXd Luu = m_Luu.block(0, 0, m_M, m_M);
        Eigen::MatrixXd Luu_inv_Kun = Luu.triangularView<Eigen::Lower>().solve(Kun);
        Eigen::MatrixXd Kuu_inv_Kun = Luu.transpose().triangularView<Eigen::Upper>().solve(Luu_inv_Kun);
        Eigen::MatrixXd mu_new = Kuu_inv_Kun.transpose() * m_mu.block(0, 0, m_M, 1);
        m_mu.block(m_M, 0, 1, 1) = mu_new;

        Eigen::MatrixXd Su = m_Su.block(0, 0, m_M, m_M);
        Eigen::MatrixXd Sun = Su * Kuu_inv_Kun;
        Eigen::MatrixXd Snn = Knn - Luu_inv_Kun.transpose() * Luu_inv_Kun + Kuu_inv_Kun.transpose() * Su * Kuu_inv_Kun;
        m_Su.block(m_M, 0, 1, m_M) = Sun.transpose();
        m_Su.block(0, m_M, m_M, 1) = Sun;
        m_Su.block(m_M, m_M, 1, 1) = Snn;
    }

    void VariationalGPR::update_mu_su(const Eigen::MatrixXd & X_new, const Eigen::MatrixXd & Y_new)
    {
        Eigen::MatrixXd Knu = kernel_->evaluate(X_new, m_inducing_point.block(0, 0, m_M, m_D));;
        Eigen::MatrixXd Ws = m_Luu.block(0, 0, m_M, m_M).triangularView<Eigen::Lower>().solve(Knu.transpose());
        
        Eigen::MatrixXd Phi_k = m_Luu.block(0, 0, m_M, m_M).transpose().triangularView<Eigen::Upper>().solve(Ws);
        Eigen::MatrixXd r_k = Y_new - Phi_k.transpose() * m_mu.block(0, 0, m_M, 1);

        Eigen::MatrixXd V = Phi_k.transpose() * m_Su.block(0, 0, m_M, m_M);
        Eigen::MatrixXd G_k;

        if (inference_method == 1)
        {
            Eigen::MatrixXd Ktt_diag = kernel_->k_diag(X_new);
            Eigen::MatrixXd Qtt_diag = Ws.array().square().colwise().sum();
            G_k = (V * Phi_k + Ktt_diag - Qtt_diag).array() + likelihood_varience;
        }
        else
        {
            G_k = (V * Phi_k).array() + likelihood_varience;
        }
        
        Eigen::MatrixXd A = m_Su.block(0, 0, m_M, m_M) * Phi_k;
        Eigen::MatrixXd L_k = A / G_k(0, 0);

        m_mu.block(0, 0, m_M, 1) = m_mu.block(0, 0, m_M, 1) + L_k * r_k(0, 0);
        m_Su.block(0, 0, m_M, m_M) = m_Su.block(0, 0, m_M, m_M) - L_k * V;
    }
}