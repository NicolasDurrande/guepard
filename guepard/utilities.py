from typing import List, Optional

from scipy.cluster.vq import kmeans

import tqdm
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
    noise_variance: Optional[float] = 0.1,
    maxiter: int = 100,
    likelihood = Optional[gpflow.likelihoods.Likelihood]
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

    if likelihood is None:
        assert noise_variance is not None
        likelihood = gpflow.likelihoods.Gaussian(variance=noise_variance)

    elbos_pre = []
    elbos_post = []
    def _create_submodel(data: RegressionData, num_inducing: int) -> SVGP:
        num_data = len(data[0])
        # num_inducing = min(num_data, num_inducing)
        # centroids = data[0][:num_inducing]
        # centroids, _ = kmeans(data[0], min(num_data, num_inducing))
        # inducing_variable = gpflow.inducing_variables.InducingPoints(centroids)
        # gpflow.set_trainable(inducing_variable, False)
        X_ = data[0][:2000]
        Y_ = data[1][:2000]
        submodel = gpflow.models.VGP(
            (X_, Y_),
            kernel=kernel,
            likelihood=likelihood,
            # inducing_variable=inducing_variable,
            mean_function=mean_function,
            # whiten=True,
        )
        if maxiter > 0:
            obj = submodel.training_loss_closure()
            elbos_pre.append(obj())
            gpflow.optimizers.scipy.Scipy().minimize(
                obj,
                submodel.trainable_variables,
                options={"disp": True, "maxiter": maxiter},
            )
            elbos_post.append(obj())
        return submodel

    print("Training submodels...")
    models = [
        _create_submodel(data, M) for data, M in tqdm.tqdm(zip(data_list, num_inducing_list), total=len(data_list))
    ]
    import matplotlib.pyplot as plt
    import tensorflow as tf
    plt.hist(tf.concat(elbos_pre, axis=0).numpy(), label="PRE")
    plt.hist(tf.concat(elbos_post, axis=0).numpy(), label="POST")
    plt.legend()
    plt.savefig("objective.png")
    return models
