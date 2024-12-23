import numpy as np
import matplotlib.pyplot as plt
import GPy

def test_plot_gpr():
    print("Loading data...")
    data = np.loadtxt("C:\\Users\\pc\\Desktop\\Personal\\Code\\KnowledgeGallery\\Machine Learning Library\\GPRcpp\\Log\\gpr.txt")

    plt.plot(data[:, 0], label = "p_w")
    plt.plot(data[:, 1], label = "p_v")
    plt.plot(data[:, 2], label = "r_w")
    plt.plot(data[:, 3], label = "r_v")
    plt.legend()
    plt.show()

def gpy_derivative_data():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]).reshape(3, 4)
    y = np.array([-0.5, 0.5, 1.5]).reshape(-1, 1)

    ard_length_scale = np.array([1, 2, 3, 4])
    rbf_kernel_1 = GPy.kern.RBF(input_dim=4, variance=0.5, lengthscale=ard_length_scale, ARD=True)

    dk_dx_1 = rbf_kernel_1.dK_dX2(x, x[0].reshape(1, 4), 0)
    dk_dx_2 = rbf_kernel_1.dK_dX2(x, x[0].reshape(1, 4), 1)
    dk_dx_3 = rbf_kernel_1.dK_dX2(x, x[0].reshape(1, 4), 2)
    dk_dx_4 = rbf_kernel_1.dK_dX2(x, x[0].reshape(1, 4), 3)
    dk_dx = np.hstack((dk_dx_1, dk_dx_2, dk_dx_3, dk_dx_4))

    """
    product dk_dx is:
           0         0         0         0
    0.178471 0.0446179 0.0198302 0.0111545
    0.253638 0.0634096  0.028182 0.0158524
    """
    print(dk_dx)

if __name__ == "__main__":
    gpy_derivative_data()
