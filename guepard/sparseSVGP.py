from typing import List, Optional

import tensorflow as tf

import gpflow
from gpflow.base import InputData, MeanAndVariance, Parameter, RegressionData
from gpflow.config import default_float
from gpflow.experimental.check_shapes import check_shapes, inherit_check_shapes
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction
from gpflow.models import ExternalDataTrainingLossMixin, GPModel
from gpflow.utilities import triangular


class SparseSVGP(GPModel, ExternalDataTrainingLossMixin):
    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        inducing_variables: List[tf.Tensor],
        q_mus: List[tf.Tensor],
        q_sqrts: List[tf.Tensor],
        *,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: int = 1,
        whiten: bool = True,
        num_data: Optional[tf.Tensor] = None,
    ):

        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_data = num_data
        self.whiten = whiten

        # init variational parameters
        Z = tf.concat(inducing_variables, axis=0)
        # self.inducing_variable: InducingVariables = gpflow.models.util.inducingpoint_wrapper(Z)
        self.inducing_variable = gpflow.inducing_variables.InducingPoints(Z)

        self.q_mu = Parameter(tf.concat(q_mus, axis=0), dtype=default_float())
        self.q_sqrts = [Parameter(q_sqrt, transform=triangular()) for q_sqrt in q_sqrts]

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points, given the variational
        pseudo-observations. The expression are the same as for a GPR model with correlated noise.
        """
        # assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        err = self.q_mu - self.mean_function(self.inducing_variable.Z)

        kmm = self.kernel(self.inducing_variable.Z)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(self.inducing_variable.Z, Xnew)
        cov_blocks = [
            tf.linalg.LinearOperatorFullMatrix(q_sqrt @ tf.transpose(q_sqrt, [0, 2, 1]))
            for q_sqrt in self.q_sqrts
        ]
        q_sigma = tf.linalg.LinearOperatorBlockDiag(cov_blocks).to_dense()
        kmm_plus_s = kmm + q_sigma[0, :, :]
        f_mean_zero, f_var = gpflow.conditionals.base_conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var

    @check_shapes(
        "return: []",
    )
    def prior_kl(self) -> tf.Tensor:
        f_mean, f_var = self.predict_f(
            self.inducing_variable.Z, full_cov=True, full_output_cov=False
        )
        f_sqrt = tf.linalg.cholesky(f_var)
        return gpflow.kullback_leiblers.prior_kl(
            self.inducing_variable, self.kernel, f_mean, f_sqrt, whiten=False
        )

    # type-ignore is because of changed method signature:
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:  # type: ignore[override]
        return self.elbo(data)

    @check_shapes(
        "return: []",
    )
    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This method is the exact same one as the SVGP class
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(X, f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl
