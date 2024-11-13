import numpy as np
from gpr_clone import my_GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]).reshape(3, 4)
y = np.array([-0.5, 0.5, 1.5])
print("----------Dataset----------")
print("x is:\n", x)
print("y is:\n", y)

const_kernel = ConstantKernel(constant_value=0.2)
const_kernel_x = const_kernel(x)
const_kernel_xy = const_kernel(x, y)
print("\n----------Constant kernel test----------")
print("const_kernel_x:\n", const_kernel_x)
print("const_kernel_xy:\n", const_kernel_xy)

white_kernel = WhiteKernel(noise_level=0.3)
white_kernel_x = white_kernel(x)
white_kernel_xy = white_kernel(x, y)
print("\n----------White kernel test----------")
print("white_kernel_x:\n", white_kernel_x)
print("white_kernel_xy:\n", white_kernel_xy)

ard_length_scale = np.array([1, 2, 3, 4])
rbf_kernel = RBF(length_scale=ard_length_scale)
rbf_kernel_x = rbf_kernel(x)
x_new = x * 2
rbf_kernel_xy = rbf_kernel(x, x_new)
print("\n----------RBF kernel test----------")
print("rbf_kernel_x:\n", rbf_kernel_x)
print("rbf_kernel_xy:\n", rbf_kernel_xy)

sum_kernel = white_kernel + rbf_kernel
sum_kernel_x = sum_kernel(x)
print("\n----------Sum kernel test----------")
print("sum_kernel_x:\n", sum_kernel_x)

product_kernel = const_kernel * rbf_kernel
product_kernel_x = product_kernel(x)
print("\n----------Product kernel test----------")
print("product_kernel_x:\n", product_kernel_x)

real_kernel = product_kernel + white_kernel
real_kernel_x = real_kernel(x)
print("\n----------Real kernel test----------")
print("real_kernel_x:\n", real_kernel_x)

gpr = my_GaussianProcessRegressor(kernel=real_kernel, optimizer=None)
gpr.fit(x, y)
mu, cov = gpr.predict(x, return_cov=True)
print("\n----------GPR predict test----------")
print("mu:\n", mu)
print("cov:\n", cov)

print("\n----------GPR normalized predict test----------")
gpr_normalized = my_GaussianProcessRegressor(kernel=real_kernel, optimizer=None, normalize_y=True)
y_multi = np.array([[1, 2], [3, 4], [5, 6]])
gpr_normalized.fit(x, y)
mu, cov = gpr_normalized.predict(x, return_cov=True)
print("mu:\n", mu)
print("cov:\n", cov)

print("\n----------Change alpha test----------")
gpr_change_alpha = my_GaussianProcessRegressor(kernel=real_kernel, optimizer=None, normalize_y=True, alpha=0.1)
y_multi = np.array([[1, 2], [3, 4], [5, 6]])
gpr_change_alpha.fit(x, y)
mu, cov = gpr_change_alpha.predict(x, return_cov=True)
print("mu:\n", mu)
print("cov:\n", cov)