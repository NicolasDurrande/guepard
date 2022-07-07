import abc
from typing import Any, Generic, List, Type, TypeVar

import tensorflow as tf
from gpflow.models import GPModel

SubModelType = TypeVar("SubModelType", bound=GPModel)


class Papl(abc.ABC, GPModel, Generic[SubModelType]):
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

        # initialise with parent class
        super().__init__(
            models[0].kernel,
            models[0].likelihood,
            models[0].mean_function,
            models[0].num_latent_gps,
        )
        self.models: List[SubModelType] = models

    def maximum_log_likelihood_objective(self, *args: Any, **kwargs: Any) -> tf.Tensor:
        raise NotImplementedError

    def training_loss(self, *args: Any, **kwargs: Any) -> tf.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _model_class(self) -> Type[SubModelType]:
        """
        Annoyingly, `SubModelType` is not available at runtime.
        By declaring it specificalyy in each subclass we can add
        this runtime check to the __init__.

        TODO: This feature will be available in the a near future release of Python -
        probably 3.12. This will make this class obsolete.
        """
        raise NotImplementedError
