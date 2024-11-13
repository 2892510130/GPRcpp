import GPy
import numpy as np

# 创建训练数据
x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]).reshape(3, 4)
y = np.array([-0.5, 0.5, 1.5]).reshape(-1, 1)

print("----------Dataset----------")
print("x is:\n", x)
print("y is:\n", y)

# Constant Kernel (常数核)
const_kernel = GPy.kern.Bias(input_dim=4, variance=0.2)  # 这里使用 Bias 核来实现常数核
const_kernel_x = const_kernel.K(x)  # 获取常数核在 x 上的值
print("\n----------Constant kernel test----------")
print("const_kernel_x:\n", const_kernel_x)

# White Kernel (Noise Kernel)
white_kernel = GPy.kern.White(input_dim=4, variance=0.3)  # 噪声核
white_kernel_x = white_kernel.K(x)
print("\n----------White kernel test----------")
print("white_kernel_x:\n", white_kernel_x)

# RBF Kernel
ard_length_scale = np.array([1, 2, 3, 4])
rbf_kernel = GPy.kern.RBF(input_dim=4, variance=1.0, lengthscale=ard_length_scale, ARD=True)  # ARD RBF 核
rbf_kernel_x = rbf_kernel.K(x)
x_new = x * 2
rbf_kernel_xy = rbf_kernel.K(x, x_new)
print("\n----------RBF kernel test----------")
print("rbf_kernel_x:\n", rbf_kernel_x)
print("rbf_kernel_xy:\n", rbf_kernel_xy)

# Sum Kernel (加法)
sum_kernel = white_kernel + rbf_kernel
sum_kernel_x = sum_kernel.K(x)
print("\n----------Sum kernel test----------")
print("sum_kernel_x:\n", sum_kernel_x)

# Product Kernel (乘法)
product_kernel = const_kernel * rbf_kernel
product_kernel_x = product_kernel.K(x)
print("\n----------Product kernel test----------")
print("product_kernel_x:\n", product_kernel_x)

# Real Kernel (加法)
real_kernel = GPy.kern.RBF(input_dim=4, variance=0.2, lengthscale=ard_length_scale, ARD=True) + white_kernel
real_kernel_x = real_kernel.K(x)
print("\n----------Real kernel test----------")
print("real_kernel_x:\n", real_kernel_x)

# 使用 GPy 进行高斯过程回归
model = GPy.models.GPRegression(x, y, kernel=real_kernel, noise_var=0.0)
print(model)

# # 打印模型信息
# print("\n----------GPR model----------")
# print(model)

# 训练并预测
# model.optimize(messages=True)
mu, var = model.predict(x, full_cov=True, include_likelihood=False) # include_likelihood=False will not include likehood variance into the predict var

print("\n----------GPR predict test----------")
print("mu:\n", mu)
print("cov:\n", var)

# 使用 normalize_y 参数进行标准化预测
model_normalized = GPy.models.GPRegression(x, y, kernel=real_kernel, normalizer=True, noise_var=0.0)
# model_normalized.optimize(messages=True)
mu_normalized, var_normalized = model_normalized.predict(x, full_cov=True, include_likelihood=False)

print("\n----------GPR normalized predict test----------")
print("mu:\n", mu_normalized)
print("cov:\n", var_normalized)

model_change_alpha = GPy.models.GPRegression(x, y, kernel=real_kernel, normalizer=True, noise_var=0.1)
# model_normalized.optimize(messages=True)
mu_change_alpha, var_change_alpha = model_change_alpha.predict(x, full_cov=True, include_likelihood=False)

print("\n----------Change alpha (noise_var) test----------")
print("mu:\n", mu_change_alpha)
print("cov:\n", var_change_alpha)