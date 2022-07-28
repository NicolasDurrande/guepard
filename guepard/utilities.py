from typing import List, Optional

from scipy.cluster.vq import kmeans

import gpflow
from gpflow.base import RegressionData
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models.gpr import GPR
from gpflow.models.svgp import SVGP


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
        submodel = SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            mean_function=mean_function,
            whiten=False,
        )
        if maxiter > 0:
            print(
                "Note that the Guepard model requires equal priors."
                "Training the models seperately will lead to different hyperparameters."
            )
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
