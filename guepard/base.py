import abc
from typing import Any, Generic, List, Type, TypeVar

import tensorflow as tf

import gpflow
from gpflow.base import InputData, MeanAndVariance
from gpflow.models import GPModel

SubModelType = TypeVar("SubModelType", bound=GPModel)


class GuepardBase(abc.ABC, Generic[SubModelType]):
    """
    Posterior Aggregation with Pseudo-Likelihood: Base class for merging submodels using the pseudo-likelihood method.
    """

    def __init__(self, models: List[SubModelType]):
        """
        :param models: A list of GPflow models with the same prior and likelihood.
        """
        # check that all models are of the same type (e.g., GPR, SVGP)
        assert all(
            [model.__class__ == self._model_class() for model in models]
        ), f"All submodels need to be of type '{self._model_class}'"
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

        self.models: List[SubModelType] = models

    @property
    def trainable_variables(self):  # type: ignore
        r = []
        for model in self.models:
            r += model.trainable_variables
        return r

    @abc.abstractmethod
    def training_loss_submodels(self, *args: Any) -> tf.Tensor:
        """
        Objective used to train the submodels
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _model_class(self) -> Type[SubModelType]:
        """
        Annoyingly, `SubModelType` is not available at runtime.
        By declaring it specifically in each subclass we can add
        this runtime check to the __init__.

        TODO: This feature will be available in the a near future release of Python -
        probably 3.12. This will make this class obsolete.
        """
        raise NotImplementedError

    # TODO: better name?
    def predict_foo(self, Xnew: InputData, full_cov: bool = False, full_output_cov = False) -> MeanAndVariance:
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
