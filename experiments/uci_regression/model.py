from typing import Tuple 
from scipy.cluster import vq

import numpy as np
import tensorflow as tf

import gpflow

import guepard


class Config:
    num_points_per_expert = 100
    kernel = "RBF"


class GuepardRegression:
    def __init__(self):
        pass

    def fit(self, X_train, Y_train):
        X_dim = X_train.shape[-1]
        num_experts = len(X_train) // Config.num_points_per_expert

        _, label_X = vq.kmeans2(X_train, num_experts, minit="points")
        data_list = [ 
            (X_train[label_X == p, :], Y_train[label_X == p, :]) for p in range(num_experts)
        ]

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
            options={"disp": True, "maxiter": 10},
        )
        self.ensemble = ensemble

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, v = self.ensemble.predict_y_marginals(X_test)
        return m.numpy(), v.numpy()
