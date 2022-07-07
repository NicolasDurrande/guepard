from typing import Any, List, Optional, Type

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.base import InputData, MeanAndVariance, RegressionData
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models import GPR
from gpflow.models.gpr import GPR

from .papl import Papl

jitter = gpflow.config.default_jitter()


def get_gpr_submodels(
    data_list: List[RegressionData],
    kernel: Kernel,
    mean_function: Optional[MeanFunction] = None,
    noise_variance: float = 0.1,
) -> List[GPR]:
    """
    Helper function to build a list of GPflow GPR submodels from a list of datasets, a GP prior and a likelihood variance.
    """
    models = [GPR(data, kernel, mean_function, noise_variance) for data in data_list]
    for m in models[1:]:
        m.likelihood = models[0].likelihood
        m.mean_function = models[0].mean_function

    return models


class GprPapl(Papl[GPR]):
    """PAPL with GPR submodels"""

    def _model_class(self) -> Type[GPR]:
        return GPR

    def predict_f_marginals(self, Xnew: InputData) -> MeanAndVariance:
        """
        Fastest method for aggregating marginal submodel predictions.
        For more accurate predictions (but with higher computational cost
        and possibly lower numerical stability), see `predict_f` method.
        :param Xnew: 2D Array or tensor corresponding to points in the input
        where we want to make prediction.
        """
        # prior predictions
        mp = self.mean_function(Xnew)[:, :, None]  # shape is [n, 1, 1]
        vp = self.kernel.K_diag(Xnew)[:, None, None]  # [n, 1, 1]

        # submodel predictons
        preds = [m.predict_f(Xnew) for m in self.models]
        Me = tf.stack([pred[0] for pred in preds], axis=2)  # [n, latent, sub]
        Ve = tf.stack([pred[1] for pred in preds], axis=2)

        # equivalent pseudo observations that would turn
        # the prior at Xnew into the expert posterior at Xnew
        pseudo_noise = vp * Ve / (vp - Ve + jitter)
        pseudo_y = mp + vp / (vp - Ve + jitter) * (Me - mp)

        # prediction
        var = 1 / (1 / vp[:, :, 0] + tf.reduce_sum(1 / pseudo_noise, axis=-1))
        mean = var * (
            mp[:, :, 0] / vp[:, :, 0] + tf.reduce_sum(pseudo_y / pseudo_noise, axis=-1)
        )

        return mean, var

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Prediction method based on the aggregation of multivariate submodel predictions.
        For more faster predictions and possibly more numerically stable predictions
        (but with lower accuracy), see the `predict_f_marginals` method.
        :param Xnew: 2D Array or tensor corresponding to points in the input
        where we want to make prediction.
        :param full_cov: Wether or not to return the full posterior covariance matrix.
        :param full_output_cov: unused
        """
        len(self.models)

        # prior distribution
        mp = self.mean_function(Xnew)[None, :, :]  # [1, N, L]
        vp = self.kernel.K(Xnew)[None, None, :, :]  # [1, L, N, N]

        # expert distributions
        preds = [m.predict_f(Xnew, full_cov=True) for m in self.models]
        Me = tf.concat([pred[0][None, :, :] for pred in preds], axis=0)  # [P, N, L]
        Ve = tf.concat(
            [pred[1][None, :, :, :] for pred in preds], axis=0
        )  # [P, L, N, N]

        # equivalent pseudo observations that would turn
        # the prior at Xnew into the expert posterior at Xnew
        Jitter = jitter * np.eye(Xnew.shape[0])[None, None, :, :]
        pseudo_noise = vp @ tf.linalg.inv(vp - Ve + Jitter) @ vp - vp
        # pseudo_noise = vp @ tf.linalg.inv(vp - Ve ) @ Ve
        # pseudo_noise = tf.linalg.inv(tf.linalg.inv(vp) - tf.linalg.inv(Ve))
        pseudo_y = (
            mp
            + vp
            @ tf.linalg.inv(vp - Ve + Jitter)
            @ tf.transpose(Me - mp, (0, 2, 1))[:, :, :, None]
        )  # [P, L, N, 1]
        # pseudo_y = mp + pseudo_noise @ tf.linalg.inv(Ve) @ (Me - mp)

        # print(np.max(np.abs(pseudo_noise - pseudo_noise_old)))
        # prediction
        var = tf.linalg.inv(
            tf.linalg.inv(vp[0, :, :])
            + tf.reduce_sum(tf.linalg.inv(pseudo_noise), axis=0)
        )
        mean = var @ (
            tf.linalg.inv(vp[0, :, :]) @ mp[0, :, :]
            + tf.reduce_sum(tf.linalg.inv(pseudo_noise) @ pseudo_y, axis=0)
        )
        return tf.transpose(mean[:, :, 0]), var

    def maximum_log_likelihood_objective(self, *args: Any, **kwargs: Any) -> tf.Tensor:
        objectives = [m.maximum_log_likelihood_objective() for m in self.models]
        return tf.reduce_sum(objectives)

    def training_loss(self, *args: Any, **kwargs: Any) -> tf.Tensor:
        return -self.maximum_log_likelihood_objective()
