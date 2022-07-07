from typing import List, Optional

import gpflow
import numpy as np
import tensorflow as tf
from scipy.cluster.vq import kmeans

from gpflow.base import InputData, MeanAndVariance, RegressionData
from gpflow.models import GPModel, SVGP
from gpflow.mean_functions import MeanFunction
from gpflow.kernels import Kernel
from gpflow.inducing_variables.inducing_variables import InducingPoints


from .papl import PAPL

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
    assert len(data_list) == len(num_inducing_list), "Please specify equal number of inducing points configs as number of datasets passed."
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
            whiten=False
        )
        gpflow.optimizers.scipy.Scipy().minimize(
            submodel.training_loss_closure(data),
            submodel.trainable_variables,
            options={"disp": False, "maxiter": 500},
        )
        return submodel

    models = [_create_submodel(data, M) for data, M in zip(data_list, num_inducing_list)]
    return models



class SparsePapl(GPModel):
    """
    Posterior Aggregation with Pseudo-Likelihood: merging submodels using the pseudo-likelihood method.

    The underlying submodels are Sparse Variational GPs (SVGPs).
    """

    def __init__(self, models: List[SVGP]):
        """
        :param models: A list of GPflow `SVGP` models with the same prior and likelihood.
        """
        # check that all models are a gpflow SVGP model
        assert all([model.__class__ for model in models] == SVGP)
        super().__init__(models)