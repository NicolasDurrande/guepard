from typing import Any, List, Optional, Type

import tensorflow as tf
from scipy.cluster.vq import kmeans

import gpflow
from gpflow.base import InputData, MeanAndVariance, RegressionData
from gpflow.experimental.check_shapes import check_shape
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models import SVGP

from .papl import Papl

jitter = gpflow.config.default_jitter()


def get_svgp_submodels(
    data_list: List[RegressionData],
    num_inducing_list: List[int],
    kernel: Kernel,
    mean_function: Optional[MeanFunction] = None,
    noise_variance: float = 0.1,
) -> List[SVGP]:
    """
    Helper function to build a list of GPflow SVGP submodels from a list of datasets, a GP prior and a likelihood variance.
    """
    assert len(data_list) == len(
        num_inducing_list
    ), "Please specify equal number of inducing points configs as number of datasets passed."
    if mean_function is None:
        mean_function = gpflow.mean_functions.Zero()

    likelihood = gpflow.likelihoods.Gaussian(variance=noise_variance)

    def _create_submodel(data: RegressionData, num_inducing: int) -> SVGP:
        num_data = len(data[0])
        centroids, _ = kmeans(data[0], min(num_data, num_inducing))
        print(centroids)
        submodel = SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=gpflow.inducing_variables.InducingPoints(centroids),
            mean_function=mean_function,
            whiten=False,
        )
        gpflow.optimizers.scipy.Scipy().minimize(
            submodel.training_loss_closure(data),
            submodel.trainable_variables,
            options={"disp": False, "maxiter": 500},
        )
        return submodel

    models = [
        _create_submodel(data, M) for data, M in zip(data_list, num_inducing_list)
    ]
    return models


class SparsePapl(Papl[SVGP]):
    """PAPL with SVGP submodels"""

    def _model_class(self) -> Type[SVGP]:
        return SVGP

    def maximum_log_likelihood_objective(self, *args: Any, **kwargs: Any) -> tf.Tensor:
        raise NotImplementedError

    def training_loss(self, *args: Any, **kwargs: Any) -> tf.Tensor:
        raise NotImplementedError

    def init_aggregate_inducing_variable(self) -> None:
        Z = check_shape(
            tf.concat(values=[m.inducing_variable.Z for m in self.models], axis=0),
            "[M, D]",
        )

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        pass
