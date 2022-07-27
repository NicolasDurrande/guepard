from itertools import zip_longest
from typing import List, Union

import tensorflow as tf

import gpflow
from gpflow.base import InputData, MeanAndVariance, RegressionData
from gpflow.models import GPModel


class EquivalentObsEnsemble(GPModel):
    """
    Posterior Aggregation with Equivalent Observation.
    """

    def __init__(self, models: List[GPModel]):
        """
        :param models: A list of GPflow models with the same prior and likelihood.
        """
        # check that all models are of the same type (e.g., GPR, SVGP)
        # assert all(
        #     [model.__class__ == self._model_class() for model in models]
        # ), f"All submodels need to be of type '{self._model_class}'"
        # check that all models have the same prior
        for model in models[1:]:
            assert (
                model.kernel == models[0].kernel
            ), "All submodels must have the same kernel"
            assert (
                model.likelihood == models[0].likelihood
            ), "All submodels must have the same likelihood"
            assert (
                model.mean_function == models[0].mean_function
            ), "All submodels must have the same mean function"
            assert (
                model.num_latent_gps == models[0].num_latent_gps
            ), "All submodels must have the same number of latent GPs"

        GPModel.__init__(
            self,
            kernel=models[0].kernel,
            likelihood=models[0].likelihood,
            mean_function=models[0].mean_function,
            num_latent_gps=models[0].num_latent_gps,
        )

        self.models: List[GPModel] = models

    @property
    def trainable_variables(self):  # type: ignore
        r = []
        for model in self.models:
            r += model.trainable_variables
        return r

    def maximum_log_likelihood_objective(self, data: List[RegressionData]) -> tf.Tensor:  # type: ignore
        [
            isinstance(m, gpflow.models.ExternalDataTrainingLossMixin)
            for m in self.models
        ]
        objectives = [m.training_loss(d) for m, d in zip_longest(self.models, data)]
        return tf.reduce_sum(objectives)

    def training_loss(
        self, data: List[Union[None, RegressionData]] = [None]
    ) -> tf.Tensor:
        external = [
            isinstance(m, gpflow.models.ExternalDataTrainingLossMixin)
            for m in self.models
        ]
        objectives = [
            m.training_loss(d) if ext else m.training_loss()
            for m, ext, d in zip_longest(self.models, external, data)
        ]
        return tf.reduce_sum(objectives)

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Prediction method based on the aggregation of multivariate submodel predictions.
        For more faster predictions and possibly more numerically stable predictions
        (but with lower accuracy), see the `predict_f_marginals` method.
        :param Xnew: 2D Array or tensor corresponding to points in the input
        where we want to make prediction.
        """
        # prior distribution
        mp = self.models[0].mean_function(Xnew)[None, :, :]  # [1, N, L]
        vp = self.models[0].kernel.K(Xnew)[None, None, :, :]  # [1, L, N, N]

        # expert distributions
        preds = [m.predict_f(Xnew, full_cov=True) for m in self.models]
        Me = tf.concat([pred[0][None, :, :] for pred in preds], axis=0)  # [P, N, L]
        Ve = tf.concat(
            [pred[1][None, :, :, :] for pred in preds], axis=0
        )  # [P, L, N, N]

        # equivalent pseudo observations that would turn
        # the prior at Xnew into the expert posterior at Xnew
        jitter = gpflow.config.default_jitter()
        Jitter = jitter * tf.eye(Xnew.shape[0], dtype=Me.dtype)[None, None, :, :]
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

        if not full_cov:
            var = tf.transpose(tf.linalg.diag_part(var))

        return tf.transpose(mean[:, :, 0]), var

    def predict_f_marginals(self, Xnew: InputData) -> MeanAndVariance:
        """
        Fastest method for aggregating marginal submodel predictions.
        For more accurate predictions (but with higher computational cost
        and possibly lower numerical stability), see `predict_f` method.
        :param Xnew: 2D Array or tensor corresponding to points in the input
        where we want to make prediction.
        """
        # prior predictions
        mp = self.models[0].mean_function(Xnew)[:, :, None]  # shape is [n, 1, 1]
        vp = self.models[0].kernel.K_diag(Xnew)[:, None, None]  # [n, 1, 1]

        # submodel predictons
        preds = [m.predict_f(Xnew) for m in self.models]
        Me = tf.stack([pred[0] for pred in preds], axis=2)  # [n, latent, sub]
        Ve = tf.stack([pred[1] for pred in preds], axis=2)

        # equivalent pseudo observations that would turn
        # the prior at Xnew into the expert posterior at Xnew
        jitter = gpflow.config.default_jitter()
        pseudo_noise = vp * Ve / (vp - Ve + jitter)
        pseudo_y = mp + vp / (vp - Ve + jitter) * (Me - mp)

        # prediction
        var = 1 / (1 / vp[:, :, 0] + tf.reduce_sum(1 / pseudo_noise, axis=-1))
        mean = var * (
            mp[:, :, 0] / vp[:, :, 0] + tf.reduce_sum(pseudo_y / pseudo_noise, axis=-1)
        )

        return mean, var
