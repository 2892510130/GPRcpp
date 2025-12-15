import GPy
import numpy as np
import re
import os

class GaussianModel:
    def __init__(self, gaussian_opts:dict):
        sparse_method = 0
        # ----- 1. Read the data and load it to numpy -----
        train_data, inducing_data = self.read_data(gaussian_opts)

        self.feature_size = gaussian_opts["input_window"] * 4 # for now the feature is [v, w, v_c, w_c] of history
        param_v, param_w = gaussian_opts["param_v"], gaussian_opts["param_w"] # 0 is gaussain var, 1 is rbf var, then is rbf length

        # ----- 2. Build the GPR for the velocity state -----
        self.ard_length_scale_v = np.array(param_v[2:])
        self.variance_v = param_v[1]
        my_kernel_v = GPy.kern.RBF(input_dim=self.feature_size, variance=self.variance_v, lengthscale=self.ard_length_scale_v, ARD=True)

        self.sparse_model_v = GPy.models.SparseGPRegression(train_data[:, :self.feature_size], train_data[:, self.feature_size].reshape(-1, 1),
                num_inducing=gaussian_opts["inducing_data_number"], Z=inducing_data[:gaussian_opts["inducing_data_number"], :],
                kernel=my_kernel_v, normalizer=True)
        self.sparse_model_v.likelihood = GPy.likelihoods.Gaussian(variance=param_v[0])
        if sparse_method == 1:
            self.sparse_model_v.inference_method = GPy.inference.latent_function_inference.FITC()
        self.sparse_model_v.parameters_changed()

        # ----- 3. Build the GPR for the rotation speed state -----
        self.ard_length_scale_w = np.array(param_w[2:])
        self.variance_w = param_w[1]
        my_kernel_w = GPy.kern.RBF(input_dim=self.feature_size, variance=self.variance_w, lengthscale=self.ard_length_scale_w, ARD=True)

        self.sparse_model_w = GPy.models.SparseGPRegression(train_data[:, :self.feature_size], train_data[:, self.feature_size + 1].reshape(-1, 1),
                num_inducing=gaussian_opts["inducing_data_number"], Z=inducing_data[gaussian_opts["inducing_data_number"]:, :],
                kernel=my_kernel_w, normalizer=True)
        self.sparse_model_w.likelihood = GPy.likelihoods.Gaussian(variance=param_w[0])
        if sparse_method == 1:
            self.sparse_model_w.inference_method = GPy.inference.latent_function_inference.FITC()
        self.sparse_model_w.parameters_changed()
    
    def read_data(self, gaussian_opts:dict):
        train_data = []
        inducing_data = []
        if os.path.exists(gaussian_opts["path"]):
            with open(gaussian_opts["path"], "r") as f:
                lines = f.readlines()

            i = 0
            for line in lines:
                if i < gaussian_opts["total_data_number"]:
                    result = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', line)
                    data = [float(d) for d in result]
                    train_data.append(data)
                elif i < gaussian_opts["total_data_number"] + 2 * gaussian_opts["inducing_data_number"]:
                    result = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', line)
                    data = [float(d) for d in result]
                    inducing_data.append(data)
                i += 1
        else:
            raise NotImplementedError("gaussain data not exist!")

        train_data = np.array(train_data)
        inducing_data = np.array(inducing_data)
        print("Traning data shape: ", train_data.shape)
        print("Inducing data shape: ", inducing_data.shape)

        return train_data, inducing_data

    def dk_dx(self, Xnew:np.ndarray):
        Xnew = Xnew.reshape(1, -1)
        mean_jac_v = np.zeros((self.sparse_model_v.Z.shape[0], self.sparse_model_v.Z.shape[1]))
        mean_jac_w = np.zeros((self.sparse_model_w.Z.shape[0], self.sparse_model_w.Z.shape[1]))

        for i in range(self.sparse_model_v.Z.shape[1]):
            mean_jac_v[:, i] = self.sparse_model_v.kern.dK_dX2(self.sparse_model_v.Z, Xnew, i).squeeze()
        for i in range(self.sparse_model_w.Z.shape[1]):
            mean_jac_w[:, i] = self.sparse_model_w.kern.dK_dX2(self.sparse_model_w.Z, Xnew, i).squeeze()
        
        return mean_jac_v, mean_jac_w

def test():
    gaussian_opts = {
        "path": "C:\\Users\\pc\\Desktop\\Personal\\Code\\GPRcpp\\Log\\py.txt",
        "total_data_number": 260,
        "inducing_data_number": 40,
        "input_window": 3,
        "lagacy": True,
        "gpr_method": 1,
        "enable_test": False,
        "param_v": [0.00026610989521133605, 15.938946986819923, 0.8658335733313588, 13.322692325094174, 6.407021934587417, 14.430933792473933, 0.8262508642621557, 12.956010458686743, 14.755220096026246, 17.347816965479478, 2.1033703167736517, 13.100624005598226, 14.435174572545298, 17.816198438786444],
        "param_w": [0.00026610989521133605, 15.938946986819923, 0.8658335733313588, 13.322692325094174, 6.407021934587417, 14.430933792473933, 0.8262508642621557, 12.956010458686743, 14.755220096026246, 17.347816965479478, 2.1033703167736517, 13.100624005598226, 14.435174572545298, 17.816198438786444],
        # "param_w": [3.5873462385031854e-05, 278.27107082447924, 179.89783828567562, 5.3518386878180575, 278.8128680361318, 37.15662167141493, 171.78692980352864, 176.27457390169434, 272.41892243231155, 258.7789715555678, 184.1016212824687, 5.335272339533333, 287.0791355720346, 220.52174051242275],
    }

    model = GaussianModel(gaussian_opts=gaussian_opts)

    x_test = model.sparse_model_v.X[:4]
    mean_v, cov_v = model.sparse_model_v.predict_noiseless(x_test, full_cov=True)
    mean_w, cov_w = model.sparse_model_w.predict_noiseless(x_test, full_cov=True)
    print("Mean result_v:\n", mean_v.squeeze())
    print("Cov result_v:\n", cov_v)
    print("Mean result_w:\n", mean_w.squeeze())
    print("Cov result_w:\n", cov_w)

    dk_dv, dk_dw = model.dk_dx(x_test)
    dmu_dv = model.sparse_model_v.normalizer.inverse_mean(dk_dv.T @ model.sparse_model_v.posterior.woodbury_vector) - model.sparse_model_v.normalizer.inverse_mean(0.0)
    dmu_dw = model.sparse_model_w.normalizer.inverse_mean(dk_dw.T @ model.sparse_model_w.posterior.woodbury_vector) - model.sparse_model_w.normalizer.inverse_mean(0.0)
    print("dmu_dv:\n", dmu_dv.squeeze())
    print("dmu_dw:\n", dmu_dw.squeeze())

if __name__ == "__main__":
    test()
