# Code based on:
# https://github.com/samcohen16/Healing-POEs-ICML/blob/master/Code/bayesian_benchmarks_modular/
# bayesian_benchmarks/models/expert_models_reg.py

# Note-worthy additions:
# - allow for multiple output dimensions
# - subclassed as GPModel


from enum import Enum
from itertools import zip_longest
from re import M
from typing import List, Optional, Union
import abc

import tensorflow as tf

import gpflow
from gpflow.base import InputData, MeanAndVariance, RegressionData
from gpflow.experimental.check_shapes import check_shape as cs
from gpflow.experimental.check_shapes import check_shapes
from gpflow.models import GPModel


class EnsembleMethods(Enum):
    """Aggregation methods"""

    POE = "PoE"
    GPOE = "gPoE"
    BCM = "BCM"
    RBCM = "rBCM"
    BARY = "Bary"


class WeightingMethods(Enum):
    """Weighting methods"""

    VAR = "Var"
    WASS = "Wasser"
    UNI = "Uniform"
    ENT = "Entropy"
    NONE = "NoWeights"


@check_shapes(
    "mu_s: [N, L, P]  # N: num data, L: num latent, P: num experts",
    "var_s: [N, L, P]",
    "power: []",
    "prior_var: [N, broadcast L, broadcast P]",
    "return: [N, L, P]",
)
def compute_weights(
    mu_s: tf.Tensor,
    var_s: tf.Tensor,
    power: float,
    weighting: WeightingMethods,
    prior_var: Optional[tf.Tensor] = None,
    softmax: bool = False,
) -> tf.Tensor:

    """Compute unnormalized weight matrix

    Inputs :
        - mu_s: predictive mean of each expert (P) at each test point (N) for each output (L)
        - var_s: predictive (marginal) variance of each expert (P) at each test point (N) for each output (L)
        - var_s: dimension: n_expert x n_test_points : predictive variance of each expert at each test point
        - power: scalar, Softmax scaling
        - weighting: weighting method
        - prior_var, prior variance
        - soft_max_wass : whether to use softmax scaling or fraction scaling

    Output :
        -- weight_matrix, dimension: n_expert x n_test_points : unnormalized weight of ith expert at jth test point
    """

    if weighting == WeightingMethods.VAR:
        return tf.math.exp(-power * var_s)

    elif weighting == WeightingMethods.WASS:
        wass = mu_s**2 + (var_s - prior_var) ** 2
        if softmax:
            return tf.math.exp(power * wass)
        else:
            return wass**power

    elif weighting == WeightingMethods.UNI:
        num_experts = tf.cast(tf.shape(mu_s)[-1], mu_s.dtype)
        return tf.ones_like(mu_s) / num_experts

    elif weighting == WeightingMethods.ENT:
        return 0.5 * (tf.math.log(prior_var) - tf.math.log(var_s))

    elif weighting == WeightingMethods.NONE.value:
        return tf.ones_like(mu_s)

    else:
        raise NotImplementedError("Unknown weighting passed to compute_weights.")



def normalize_weights(weight_matrix):
    sum_weights = tf.reduce_sum(weight_matrix, axis=-1, keepdims=True)
    weight_matrix = weight_matrix / sum_weights
    return weight_matrix


class Ensemble(GPModel):
    """
    Implements a range of Ensemble GP models.
    """

    def __init__(
        self,
        models: List[GPModel],
        method: EnsembleMethods,
        weighting: WeightingMethods,
        power: float = 8.0,
    ):
        """
        :param models: A list of GPflow models with the same prior and likelihood.
        """
        # check that all models are of the same type (e.g., GPR, SVGP)
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

        GPModel.__init__(
            self,
            kernel=models[0].kernel,
            likelihood=models[0].likelihood,
            mean_function=models[0].mean_function,
            num_latent_gps=models[0].num_latent_gps,
        )
        self.method = method
        self.weighting = weighting
        self.power = power
        self.models: List[GPModel] = models

    @property
    def trainable_variables(self):  # type: ignore
        r = []
        for model in self.models:
            r += model.trainable_variables
        return r

    def maximum_log_likelihood_objective(self, data: List[RegressionData]) -> tf.Tensor:  # type: ignore
        [
            isinstance(m, gpflow.models.ExternalDataTrainingLossMixin)
            for m in self.models
        ]
        objectives = [m.training_loss(d) for m, d in zip_longest(self.models, data)]
        return tf.reduce_sum(objectives)

    def training_loss(
        self, data: List[Union[None, RegressionData]] = [None]
    ) -> tf.Tensor:
        external = [
            isinstance(m, gpflow.models.ExternalDataTrainingLossMixin)
            for m in self.models
        ]
        objectives = [
            m.training_loss(d) if ext else m.training_loss()
            for m, ext, d in zip_longest(self.models, external, data)
        ]
        return tf.reduce_sum(objectives)

    @check_shapes(
        "Xnew: [N, D]",
        "return[0]: [N, broadcast L]",
        "return[1]: [N, broadcast L]",
    )
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Prediction method based on the aggregation of multivariate submodel predictions.
        For more faster predictions and possibly more numerically stable predictions
        (but with lower accuracy), see the `predict_f_marginals` method.
        :param Xnew: 2D Array or tensor corresponding to points in the input
        where we want to make prediction.
        """
        assert not full_cov
        assert not full_output_cov
        b = "broadcast"

        # prior distribution
        vp = cs(self.models[0].kernel.K_diag(Xnew)[:, None, None], f"[N, {b} L, {b} P]")

        # expert distributions
        # P: number of experts
        preds = [m.predict_f(Xnew) for m in self.models]
        Me = cs(
            tf.stack([pred[0] for pred in preds], axis=2), f"[N, {b} L, P]"
        )  # [n, latent, sub]
        Ve = cs(tf.stack([pred[1] for pred in preds], axis=2), f"[N, {b} L, P]")

        # Compute individual precisions - dim: n_experts x n_test_points
        prec_s = cs(1.0 / Ve, f"[N, {b} L, P]")

        weight_matrix = cs(
            compute_weights(Me, Ve, self.power, self.weighting, vp), f"[N, {b} L, P]"
        )

        # For all DgPs, normalized weights of experts requiring normalized weights and compute the aggegated local precisions
        if self.method == EnsembleMethods.POE.value:
            prec = cs(tf.reduce_sum(prec_s, axis=-1), f"[N, {b} L]")

        elif self.method == EnsembleMethods.GPOE:
            # weight_matrix = tf.linalg.normalize(weight_matrix, ord=1, axis=-1)
            weight_matrix = normalize_weights(weight_matrix)
            prec = tf.reduce_sum(weight_matrix * prec_s, axis=-1)

        elif self.method == EnsembleMethods.BCM:
            num_experts = tf.cast(tf.shape(vp)[-1], vp.dtype)
            prec = tf.reduce_sum(prec_s, axis=-1) + (1.0 - num_experts) / vp[..., 0]

        elif self.method == EnsembleMethods.RBCM:
            prec = (
                tf.reduce_sum(weight_matrix * prec_s, axis=-1)
                + (1.0 - tf.reduce_sum(weight_matrix, axis=-1)) / vp[..., 0]
            )

        # Compute the aggregated predictive means and variance of the barycenter
        if self.method == EnsembleMethods.BARY:
            # weight_matrix = tf.linalg.normalize(weight_matrix, ord=1, axis=-1)
            weight_matrix = normalize_weights(weight_matrix)
            mu = tf.reduce_sum(weight_matrix * Me, axis=-1)
            var = tf.reduce_sum(weight_matrix * Ve, axis=-1)
        # For all DgPs compute the aggregated predictive means and variance
        else:
            prec = cs(prec, f"[N, {b} L]")
            var = 1.0 / prec
            mu = var * tf.reduce_sum(weight_matrix * prec_s * Me, axis=-1)

        return mu, var

class GPEnsemble(GPModel, metaclass=abc.ABCMeta):
    """
    Base class for GP ensembles.
    """

    def __init__(
        self,
        models: List[GPModel],
    ):
        """
        :param models: A list of GPflow models with the same prior and likelihood.
        """
        # check that all models are of the same type (e.g., GPR, SVGP)
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

        GPModel.__init__(
            self,
            kernel=models[0].kernel,
            likelihood=models[0].likelihood,
            mean_function=models[0].mean_function,
            num_latent_gps=models[0].num_latent_gps,
        )

        self.models: List[GPModel] = models

    @property
    def trainable_variables(self):  # type: ignore
        r = []
        for model in self.models:
            r += model.trainable_variables
        return r

    def maximum_log_likelihood_objective(self, data: List[RegressionData]) -> tf.Tensor:  # type: ignore
        [
            isinstance(m, gpflow.models.ExternalDataTrainingLossMixin)
            for m in self.models
        ]
        objectives = [m.training_loss(d) for m, d in zip_longest(self.models, data)]
        return tf.reduce_sum(objectives)

    def training_loss(
        self, data: List[Union[None, RegressionData]] = [None]
    ) -> tf.Tensor:
        external = [
            isinstance(m, gpflow.models.ExternalDataTrainingLossMixin)
            for m in self.models
        ]
        objectives = [
            m.training_loss(d) if ext else m.training_loss()
            for m, ext, d in zip_longest(self.models, external, data)
        ]
        return tf.reduce_sum(objectives)

    @abc.abstractclassmethod
    @check_shapes(
        "Xnew: [N, D]",
        "return[0]: [N, broadcast L]",
        "return[1]: [N, broadcast L]",
    )
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        raise NotImplementedError


class NestedGP(GPEnsemble):
    """
    Implements the Nested GP predictor (aka NPAE).
    """

    @check_shapes(
        "Xnew: [N, D]",
        "return[0]: [N, broadcast L]",
        "return[1]: [N, broadcast L]",
    )
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:

        assert not full_output_cov

        X = tf.concat([m.data[0] for m in self.models], axis=0) # [n, d]
        Y = tf.concat([m.data[1] for m in self.models], axis=0) # [n, 1]

        ki_list = [self.kernel(Xnew, m.data[0]) for m in self.models]  # elements are [q, ni] 
        Ki_list = [self.kernel(m.data[0]) + self.likelihood.variance * tf.eye(m.data[0].shape[0], dtype=tf.float64) for m in self.models] # elements are [ni, ni]

        Alpha_list = [ki @ tf.linalg.inv(Ki) for ki, Ki in zip(ki_list, Ki_list)]   # elements are [q, ni]
        Alpha_ops = [tf.linalg.LinearOperatorFullMatrix(a[:, None, :]) for a in Alpha_list] 
        Alpha = tf.linalg.LinearOperatorBlockDiag(Alpha_ops)  # [q, p, n]

        Mx = Alpha.matmul(Y) # [q, p, 1]
 
        kM_list = [tf.reduce_sum(alpha * ki, axis=1, keepdims=True) for alpha, ki in zip(Alpha_list, ki_list)] # elements are [q, 1] 
        kM = tf.stack(kM_list, axis=2) # [q, 1, p] 

        K = tf.expand_dims(self.kernel(X), axis=0)
        AM = Alpha.to_dense()
        KM =   AM @ K @ tf.transpose(AM, perm=[0,2,1])
        #KM = Alpha.matmul(Alpha.matmul(K), adjoint_arg=True)  # Should be more efficient numerically, but does not return the right values at the moment!

        Alpha2 = kM @ tf.linalg.inv(KM)  # [q, 1, p]
        mean =  Alpha2 @ Mx # [q, 1, 1]

        var_correction = Alpha2 @ tf.transpose(kM, perm=[0,2,1]) # [q, 1, 1]
        var = self.kernel.K_diag(Xnew)[:, None] - var_correction[:, :, 0] # [q, 1]

        return mean[:, 0, :], var