from typing import Tuple, List, final
from scipy.cluster import vq

import numpy as np
import tensorflow as tf

import gpflow

import guepard
from guepard.baselines import EnsembleMethods, WeightingMethods, Ensemble


class Config:
    num_points_per_expert_small = 100
    num_points_per_expert_large = 1000
    threshold_small_to_large = 5_000
    kernel = "RBF"
    maxiter = 100


__all__ = [
    "GuepardRegression",
    "gPoE_unif",
    "gPoE_var",
    "rBCM_entr",
    "BAR_var",
]


def _get_experts_and_datasets(X_train, Y_train, ard=True) -> Tuple[List[gpflow.models.GPR], List[gpflow.base.RegressionData]]:
    """Uses kmeans to create subsets of data and builds the GPR models."""
    num_data, X_dim = X_train.shape
    if num_data > Config.threshold_small_to_large:
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
    print("Min point per expert", min([len(t[0]) for t in data_list]))

    if Config.kernel == "RBF":
        ells = np.ones((X_dim,)) if ard else 0.1
        # kernel = gpflow.kernels.SquaredExponential(lengthscales=np.ones((X_dim,)) * 1e-1)
        kernel = gpflow.kernels.SquaredExponential(lengthscales=ells)
        gpflow.set_trainable(kernel.variance, False)
    else:
        raise NotImplementedError(f"Unknown kernel {Config.kernel}")

    experts = guepard.utilities.get_gpr_submodels(
        data_list, kernel, mean_function=None, noise_variance=1e-1
    )
    return experts, data_list


class GuepardRegression:

    def fit(self, X_train, Y_train):
        d = X_train.shape[-1]
        ard = d < 10
        submodels, data_list = _get_experts_and_datasets(X_train, Y_train, ard=ard)
        ensemble = guepard.EquivalentObsEnsemble(submodels)
        try:
            gpflow.optimizers.scipy.Scipy().minimize(
                tf.function(lambda: ensemble.training_loss(data_list)),
                ensemble.trainable_variables,
                options={"disp": False, "maxiter": Config.maxiter},
            )
        except Exception as e:
            print(('!' * 10) + "Exception during optimization")
            raise e
        finally:
            print("variance", ensemble.kernel.variance)
            print("lengthscales", ensemble.kernel.lengthscales)
            print("likelihood.variance", ensemble.likelihood.variance)
        self.ensemble = ensemble

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, v = self.ensemble.predict_y_marginals(X_test)
        return m.numpy(), v.numpy()


class _BaselineEnsemble:

    def __init__(self, method: EnsembleMethods, weighting: WeightingMethods):
        self._method = method
        self._weighting = weighting

    def fit(self, X_train, Y_train):
        experts, data_list = _get_experts_and_datasets(X_train, Y_train, ard=False)

        ensemble = Ensemble(experts, self._method, self._weighting)

        gpflow.optimizers.scipy.Scipy().minimize(
            tf.function(lambda: ensemble.training_loss(data_list)),
            ensemble.trainable_variables,
            options={"disp": True, "maxiter": Config.maxiter},
        )
        self.ensemble = ensemble

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, v = self.ensemble.predict_y(X_test)
        return m.numpy(), v.numpy()


gPoE_unif = lambda: _BaselineEnsemble(EnsembleMethods.GPOE, WeightingMethods.UNI)
gPoE_var = lambda: _BaselineEnsemble(EnsembleMethods.GPOE, WeightingMethods.VAR)
rBCM_entr = lambda: _BaselineEnsemble(EnsembleMethods.RBCM, WeightingMethods.ENT)
BAR_var = lambda: _BaselineEnsemble(EnsembleMethods.BARY, WeightingMethods.VAR)