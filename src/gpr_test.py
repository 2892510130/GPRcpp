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
    param = [0.11018349064219966, 2.7818099637910776, 3.3299838906955412, 5.806770560166751, 0.3458158499284298, 9.116170874885317, 3.0008849889503635, 6.419472222703577, 7.906210847681567, 16.150254238054373, 0.2920175102968179, 0.5055500569589852, 0.5951417958654771, 18.3486568783593]
    ard_length_scale = np.array(param[2:])
    rbf_kernel_1 = GPy.kern.RBF(input_dim=4, variance=param[1], lengthscale=ard_length_scale, ARD=True)

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
