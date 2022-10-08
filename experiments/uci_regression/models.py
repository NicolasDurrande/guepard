from typing import Tuple 
from scipy.cluster import vq

import numpy as np
import tensorflow as tf

import gpflow

import guepard


class Config:
    num_points_per_expert_small = 100
    num_points_per_expert_large = 1000
    large_threshold = 10_000
    kernel = "RBF"
    maxiter = 100


class GuepardRegression:
    def __init__(self):
        pass

    def fit(self, X_train, Y_train):
        num_data, X_dim = X_train.shape
        if num_data > Config.large_threshold:
            num_points_per_expert = Config.num_points_per_expert_large
        else:
            num_points_per_expert = Config.num_points_per_expert_small
        
        num_experts = len(X_train) // num_points_per_expert

        _, label_X = vq.kmeans2(X_train, num_experts, minit="points")
        data_list = [ 
            (X_train[label_X == p, :], Y_train[label_X == p, :]) for p in range(num_experts)
        ]
        print("Num experts", num_experts)
        print("Max point per expert", max([len(t[0]) for t in data_list]))

        if Config.kernel == "RBF":
            kernel = gpflow.kernels.SquaredExponential(lengthscales=np.ones((X_dim,)) * 1e-1)
        else:
            raise NotImplementedError(f"Unknown kernel {Config.kernel}")

        submodels = guepard.utilities.get_gpr_submodels(
            data_list, kernel, mean_function=None, noise_variance=1e-1
        )
        ensemble = guepard.EquivalentObsEnsemble(submodels)
        gpflow.optimizers.scipy.Scipy().minimize(
            tf.function(lambda: ensemble.training_loss(data_list)),
            ensemble.trainable_variables,
            options={"disp": True, "maxiter": Config.maxiter},
        )
        self.ensemble = ensemble

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, v = self.ensemble.predict_y_marginals(X_test)
        return m.numpy(), v.numpy()



from guepard.baselines import EnsembleMethods, WeightingMethods, Ensemble

class _BaselineEnsemble:

    def __init__(self, ):
        pass

    def fit(self, X_train, Y_train):
        num_data, X_dim = X_train.shape
        if num_data > Config.large_threshold:
            num_points_per_expert = Config.num_points_per_expert_large
        else:
            num_points_per_expert = Config.num_points_per_expert_small
        
        num_experts = len(X_train) // num_points_per_expert

        _, label_X = vq.kmeans2(X_train, num_experts, minit="points")
        data_list = [ 
            (X_train[label_X == p, :], Y_train[label_X == p, :]) for p in range(num_experts)
        ]
        print("Num experts", num_experts)
        print("Max point per expert", max([len(t[0]) for t in data_list]))

        if Config.kernel == "RBF":
            kernel = gpflow.kernels.SquaredExponential(lengthscales=np.ones((X_dim,)) * 1e-1)
        else:
            raise NotImplementedError(f"Unknown kernel {Config.kernel}")

        submodels = guepard.utilities.get_gpr_submodels(
            data_list, kernel, mean_function=None, noise_variance=1e-1
        )
        ensemble = guepard.EquivalentObsEnsemble(submodels)
        gpflow.optimizers.scipy.Scipy().minimize(
            tf.function(lambda: ensemble.training_loss(data_list)),
            ensemble.trainable_variables,
            options={"disp": True, "maxiter": Config.maxiter},
        )
        self.ensemble = ensemble

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, v = self.ensemble.predict_y_marginals(X_test)
        return m.numpy(), v.numpy()
