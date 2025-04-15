import GPy
import numpy as np

print("\n----------Big data test----------")
big_data_path = "C:\\Users\\pc\\Desktop\\Personal\\Code\\GPRcpp\\Log\\gazebo1.txt"
total_data_number = 291
feature_size = 12
inducing_data_number = 40

train_data = []
inducing_data = []

import os
import re
if os.path.exists(big_data_path):
    with open(big_data_path, "r") as f:
        lines = f.readlines()
    
    train_data = []
    i = 0
    for line in lines:
        if i < total_data_number:
            result = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', line)
            data = [float(d) for d in result]
            train_data.append(data)
        elif i < total_data_number + 2 * inducing_data_number:
            result = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', line)
            data = [float(d) for d in result]
            inducing_data.append(data)
        i += 1
    
train_data = np.array(train_data)
inducing_data = np.array(inducing_data)
print(train_data.shape)
print(inducing_data.shape)

ard_length_scale = np.array([100.017, 44.7549, 14.9609, 54.1276, 0.080009, 48.3174, 100.014, 100.496, 45.3872, 68.301, 16.327, 57.2148])
rbf_kernel_1 = GPy.kern.RBF(input_dim=12, variance=1.41831, lengthscale=ard_length_scale, ARD=True)
rbf_kernel_2 = GPy.kern.RBF(input_dim=12, variance=1.21955e-11, lengthscale=0.0101869)
my_kernel = rbf_kernel_1 + rbf_kernel_2

model = GPy.models.GPRegression(train_data[:, :12], train_data[:, 13].reshape(-1, 1), kernel=my_kernel, normalizer=True, noise_var=0.0)
# model.inference_method = GPy.inference.latent_function_inference.FITC()
# model.parameters_changed()
# model.optimize()
# print(model)
mu, var = model.predict(train_data[:, :12], full_cov=True, include_likelihood=False)
print("exact mu:\n", mu[:4].T)
print("exact cov:\n", var[:4, :4])

sparse_model = GPy.models.SparseGPRegression(train_data[:, :12], train_data[:, 13].reshape(-1, 1), 
                num_inducing=inducing_data_number, Z=inducing_data[inducing_data_number:, :],kernel=my_kernel, normalizer=True)
sparse_model.likelihood = GPy.likelihoods.Gaussian(variance=0.33669)
sparse_model.parameters_changed()
mu, var = sparse_model.predict(train_data[:, :12], full_cov=True, include_likelihood=False)
print("sparse mu:\n", mu[:4].T)
print("sparse cov:\n", var[:4, :4])


mu, var2 = sparse_model.predict(train_data[:, :12], full_cov=True)
print("Add the likelihood\n", mu[:4].T)
print("Add the likelihood\n", var2[:4, :4])
print(sparse_model.likelihood.variance)
