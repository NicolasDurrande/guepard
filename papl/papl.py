from typing import List

import gpflow
import tensorflow as tf
from gpflow.models import GPModel

jitter = gpflow.config.default_jitter()


class PAPL(GPModel):
    def __init__(self, models: List[GPModel]):

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

        # initialise with parent class
        super().__init__(
            models[0].kernel,
            models[0].likelihood,
            models[0].mean_function,
            models[0].num_latent_gps,
        )
        self.models = models

    def _predict_f_marginals(self, x):
        # prior predictions
        mp = self.mean_function(x)[:, :, None]  # shape is [n, 1, 1]
        vp = self.kernel.K_diag(x)[:, None, None]  # [n, 1, 1]

        # submodel predictons
        preds = [m.predict_f(x) for m in self.models]
        Me = tf.stack([pred[0] for pred in preds], axis=2)  # [n, latent, sub]
        Ve = tf.stack([pred[1] for pred in preds], axis=2)

        # equivalent pseudo observations that would turn
        # the prior at x into the expert posterior at x
        pseudo_noise = vp * Ve / (vp - Ve + jitter)
        pseudo_y = mp + vp / (vp - Ve + jitter) * (Me - mp)

        # prediction
        var = 1 / (1 / vp[:, :, 0] + tf.reduce_sum(1 / pseudo_noise, axis=-1))
        mean = var * (
            mp[:, :, 0] / vp[:, :, 0] + tf.reduce_sum(pseudo_y / pseudo_noise, axis=-1)
        )

        return mean, var

    def maximum_log_likelihood_objective(self):
        pass

    def predict_f(self, Xnew):
        pass
