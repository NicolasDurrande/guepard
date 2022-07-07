from typing import List, Optional, Type

import tensorflow as tf
from scipy.cluster.vq import kmeans

import gpflow
from gpflow.base import RegressionData
from gpflow.experimental.check_shapes import check_shape, check_shapes
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
    maxiter: int = 100,
) -> List[SVGP]:
    """
    Helper function to build a list of GPflow SVGP submodels from a list of datasets, a GP prior and a likelihood variance.

    :param maxiter: number of training iterations. If set to -1, no training will occur.
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
        inducing_variable = gpflow.inducing_variables.InducingPoints(centroids)
        gpflow.set_trainable(inducing_variable, False)
        print(centroids)
        submodel = SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            mean_function=mean_function,
            whiten=False,
        )
        if maxiter > 0:
            gpflow.optimizers.scipy.Scipy().minimize(
                submodel.training_loss_closure(data),
                submodel.trainable_variables,
                options={"disp": False, "maxiter": maxiter},
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

    @check_shapes()
    def get_ensemble_svgp(self) -> SVGP:
        # total_num_inducing = sum((len(m.inducing_variable) for m in self.models))
        # input_dim = self.models[0].inducing_variable.Z.shape[-1]
        Z = check_shape(
            tf.concat(values=[m.inducing_variable.Z for m in self.models], axis=0),
            "[M, D]",
        )
        iv = gpflow.inducing_variables.InducingPoints(Z)
        q_mu, q_sqrt = self.predict_foo(Z)
        return SVGP(
            self.models[0].kernel,
            self.models[0].likelihood,
            inducing_variable=iv,
            mean_function=self.models[0].mean_function,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            whiten=False,
        )

    def training_loss_submodels(self, data: List[RegressionData]) -> tf.Tensor:  # type: ignore
        """
        Objective used to train the submodels
        """
        objectives = [m.training_loss(d) for m, d in zip(self.models, data)]
        return tf.reduce_sum(objectives)

    @check_shapes()
    def init_aggregate_inducing_variable(self) -> None:
        Z = check_shape(
            tf.concat(values=[m.inducing_variable.Z for m in self.models], axis=0),
            "[M, D]",
        )
        print(Z)
