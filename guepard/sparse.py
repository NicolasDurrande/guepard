from typing import List, Optional, Type

import gpflow
from gpflow.base import RegressionData
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models import SVGP
from scipy.cluster.vq import kmeans

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

    def _create_submodel(data, num_inducing):
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

    def init_aggregate_inducing_variable(self):
        pass
        # self.models[0].
        # Z = tf.concat([
        #     model.in for model in self.models
        # ])
