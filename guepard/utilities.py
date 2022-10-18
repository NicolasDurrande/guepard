from typing import List, Optional, Tuple

import numpy as np
import tqdm
from tensorflow import Tensor

import gpflow
from gpflow.base import RegressionData
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction
from gpflow.models import GPR, SVGP

from .equivalentobs import EquivalentObsEnsemble


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
    likelihood: Likelihood = gpflow.likelihoods.Gaussian(variance=0.1),
    mean_function: Optional[MeanFunction] = None,
    noise_variance: Optional[float] = 0.1,
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

    if likelihood is None:
        assert noise_variance is not None
        likelihood = gpflow.likelihoods.Gaussian(variance=noise_variance)

    elbos_pre = []
    elbos_post = []

    def _create_submodel(data: RegressionData, num_inducing: int) -> SVGP:
        num_data = len(data[0])
        num_inducing = min(num_data, num_inducing)
        centroids = data[0][:num_inducing]
        # centroids, _ = kmeans(data[0], min(num_data, num_inducing))
        inducing_variable = gpflow.inducing_variables.InducingPoints(centroids)
        gpflow.set_trainable(inducing_variable, False)
        # X_ = data[0][:num_inducing]
        # Y_ = data[1][:num_inducing]
        submodel = gpflow.models.SVGP(
            # (X_, Y_),
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            mean_function=mean_function,
            # whiten=True,
        )

        if maxiter > 0:
            # obj = submodel.training_loss_closure()
            obj = submodel.training_loss_closure(data)
            elbos_pre.append(obj())
            gpflow.optimizers.scipy.Scipy().minimize(
                obj,
                submodel.trainable_variables,
                options={"disp": False, "maxiter": maxiter},
            )
            elbos_post.append(obj())
        return submodel

    print("Training submodels...")
    models = [
        _create_submodel(data, M)
        for data, M in tqdm.tqdm(
            zip(data_list, num_inducing_list), total=len(data_list)
        )
    ]

    # plt.hist(tf.concat(elbos_pre, axis=0).numpy(), label="PRE")
    # plt.hist(tf.concat(elbos_post, axis=0).numpy(), label="POST")
    # plt.legend()
    # plt.savefig("objective.png")
    return models


def init_ssvgp_with_ensemble(
    M: List[SVGP], add_jitter: Optional[bool] = False
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    m_ens = EquivalentObsEnsemble(M)
    Zs = [m.inducing_variable.Z for m in M]
    ind_Zi = [0] + list(np.cumsum([Zi.shape[0] for Zi in Zs]))

    Z = np.vstack(Zs)
    num_inducing = Z.shape[0]

    q_m, q_v = m_ens.predict_f(Z, full_cov=True)

    inv_noise = np.linalg.inv(q_v[0]) - np.linalg.inv(m_ens.kernel(Z))
    q_sigmas = [
        np.linalg.inv(inv_noise[i:j, i:j]) for i, j in zip(ind_Zi[:-1], ind_Zi[1:])
    ]

    if add_jitter:
        jitter = gpflow.config.default_jitter()
        q_sigmas = [q_sigma + jitter * np.eye(q_sigma.shape[0]) for q_sigma in q_sigmas]

    q_sqrts = [np.linalg.cholesky(q_sigma)[None, :, :] for q_sigma in q_sigmas]

    q_mu = (
        np.eye(num_inducing)
        + np.linalg.inv(m_ens.kernel(Z) @ np.linalg.inv(q_v[0]) - np.eye(num_inducing))
    ) @ q_m
    q_mus = [q_mu[i:j] for i, j in zip(ind_Zi[:-1], ind_Zi[1:])]

    return Zs, q_mus, q_sqrts
