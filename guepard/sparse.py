from typing import Any, List, Optional, Tuple, Type

import tensorflow as tf
from scipy.cluster.vq import kmeans

import gpflow
from gpflow.base import RegressionData
from gpflow.experimental.check_shapes import check_shape, check_shapes
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models import SVGP
from gpflow.models.svgp import SVGP_deprecated

from .base import GuepardBase

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


class SparseGuepard(GuepardBase[SVGP], SVGP_deprecated):
    def __init__(self, models: List[SVGP]):
        GuepardBase.__init__(self, models)

        Z = tf.concat(values=[m.inducing_variable.Z for m in models], axis=0)

        iv = gpflow.inducing_variables.InducingPoints(Z)
        SVGP_deprecated.__init__(
            self,
            models[0].kernel,
            models[0].likelihood,
            inducing_variable=iv,
            mean_function=models[0].mean_function,
            whiten=False,
        )

    def _model_class(self) -> Type[SVGP]:
        return SVGP

    def training_loss_submodels(self, data: List[RegressionData]) -> tf.Tensor:  # type: ignore
        """
        Objective used to train the submodels
        """
        objectives = [m.training_loss(d) for m, d in zip(self.models, data)]
        return tf.reduce_sum(objectives)

    def get_qmu_qsqrt(self) -> Tuple[tf.Tensor, tf.Tensor]:
        q_mu, q_var = self.predict_foo(self.inducing_variable.Z)
        jitter = 1e-6
        q_sqrt = tf.linalg.cholesky(
            q_var
            + tf.eye(self.inducing_variable.Z.shape[0], dtype=q_var.dtype)[None]
            * jitter
        )
        return q_mu, q_sqrt

    @property
    def q_mu(self) -> tf.Tensor:  # type: ignore[override]
        return self.get_qmu_qsqrt()[0]

    @property
    def q_sqrt(self) -> tf.Tensor:  # type: ignore[override]
        return self.get_qmu_qsqrt()[1]

    @q_mu.setter  # type: ignore[attr-defined, no-redef]
    def q_mu(self, _: Any) -> None:
        """
        q_mu can not be set as it is derived from the submodels.
        We don' throw an error because the super class SVGP
        tries to set a value for q_mu at initialisation.
        """

    @q_sqrt.setter  # type: ignore[attr-defined, no-redef]
    def q_sqrt(self, _: Any) -> None:
        """
        q_sqrt can not be set as it is derived from the submodels.
        We don' throw an error because the super class SVGP
        tries to set a value for q_sqrt at initialisation.
        """

    @check_shapes()
    def get_fully_parameterized_svgp(self) -> SVGP:
        """
        Returns 'full' SVGP (i.e. a SVGP with a full parameterisation of q_mu and q_sqrt).
        The q_mu and q_sqrt are intiatialised through the Guepard parameterisation.
        Optimisation of this models breaks the Guepard parameterisation and should lead
        to the optimal ELBO.
        """
        Z = check_shape(
            tf.concat(values=[m.inducing_variable.Z for m in self.models], axis=0),
            "[M, D]",
        )
        iv = gpflow.inducing_variables.InducingPoints(Z)
        q_mu, q_sqrt = self.predict_foo(Z, full_cov=True)
        # q_mu, q_sqrt = self.get_qmu_qsqrt() #TODO: this seems the right thing to do, but results in a Cholesky error...

        return SVGP(
            self.models[0].kernel,
            self.models[0].likelihood,
            inducing_variable=iv,
            mean_function=self.models[0].mean_function,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            whiten=False,
        )
