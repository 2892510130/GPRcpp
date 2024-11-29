import GPy
import numpy as np

x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]).reshape(3, 4)
y = np.array([-0.5, 0.5, 1.5]).reshape(-1, 1)

ard_length_scale = np.array([1, 2, 3, 4])
rbf_kernel_1 = GPy.kern.RBF(input_dim=4, variance=0.5, lengthscale=ard_length_scale, ARD=True)
rbf_kernel_2 = GPy.kern.RBF(input_dim=4, variance=2.0, lengthscale=0.5)
my_kernel = rbf_kernel_1 + rbf_kernel_2

model = GPy.models.SparseGPRegression(x, y, kernel=my_kernel, num_inducing = 2, Z = x[:2], normalizer=False)
model.inference_method = GPy.inference.latent_function_inference.FITC()
model.parameters_changed()
mu, var = model.predict(x, full_cov=True, include_likelihood=False)

print("mu:\n", mu)
print("cov:\n", var)