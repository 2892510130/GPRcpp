# GPRcpp
This is a simple implemention of Gaussian Process Regression in C++. Most of the code is translate from sklearn library and GPy library.
See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html and https://gpy.readthedocs.io/en/deploy/ for reference.

## Question
LDLT do no work, because of the P matrix. In LDLT, we get P^T * L * D * L^T * P = A, here L is lower triangular matrix. Though LLT can not work well in some platform like aarch64, and LDLT can, but we are stuck with LLT.