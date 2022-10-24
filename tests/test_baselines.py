import numpy as np
import pytest

import gpflow

from guepard.baselines import Ensemble, EnsembleMethods, NestedGP, WeightingMethods
from guepard.utilities import get_gpr_submodels

np.random.seed(123456)


def get_data(num_data):
    X = np.linspace(0, 1, num_data)[:, None]
    F = gpflow.models.GPR(
        (np.c_[-10.0], np.c_[0.0]), KERNEL, noise_variance=NOISE_VAR
    ).predict_f_samples(X)
    Y = F + np.random.randn(*F.numpy().shape) * NOISE_VAR**0.5
    return X, Y


NOISE_VAR = 1e-3
NUM_DATA = 100
KERNEL = gpflow.kernels.SquaredExponential(lengthscales=0.1)
(X, Y) = get_data(NUM_DATA)

num_split = 10
x_list = np.array_split(X, num_split)  # list of num_split np.array
y_list = np.array_split(Y, num_split)

# make submodels and aggregate them
datasets = list(zip(x_list, y_list))
submodels = get_gpr_submodels(
    datasets, KERNEL, mean_function=None, noise_variance=NOISE_VAR
)  # list of num_split GPR models


@pytest.mark.parametrize("method", EnsembleMethods)
@pytest.mark.parametrize("weighting", WeightingMethods)
def test_predict_f_smoke(method, weighting):
    """
    Baseline methods are too inaccurate to have a meaningful comparison with the
    GPR ground truth. This is just a smoke-test that compares shapes of outputs.
    """
    # make submodels and aggregate them
    m_agg = Ensemble(submodels, method, weighting)

    # make a GPR model as baseline
    m_gpr = gpflow.models.GPR((X, Y), KERNEL, noise_variance=NOISE_VAR)

    # Check shapes of output matches the GPflow convention
    mean_agg, var_agg = m_agg.predict_f(X, full_cov=False)
    mean_gpr, var_gpr = m_gpr.predict_f(X, full_cov=False)

    np.testing.assert_array_almost_equal(
        mean_agg.shape,
        mean_gpr.shape,
        err_msg=f"mismatch between the ensemble predict_f mean shape and the GPflow convention",
    )

    np.testing.assert_array_almost_equal(
        var_agg.shape,
        var_gpr.shape,
        err_msg=f"mismatch between the {method, weighting} predict_f variance shape and the GPflow convention",
    )


def test_predict_f_NestedGP():

    # make submodels and aggregate them
    m_agg = NestedGP(submodels)

    # make a GPR model as baseline
    m_gpr = gpflow.models.GPR((X, Y), KERNEL, noise_variance=NOISE_VAR)

    # Check shapes of output matches the GPflow convention
    mean_agg, var_agg = m_agg.predict_f(X, full_cov=False)
    mean_gpr, var_gpr = m_gpr.predict_f(X, full_cov=False)

    np.testing.assert_array_almost_equal(
        mean_agg.shape,
        mean_gpr.shape,
        err_msg="mismatch between the NestedGP predict_f mean shape and the GPflow convention",
    )

    np.testing.assert_array_almost_equal(
        var_agg.shape,
        var_gpr.shape,
        err_msg="mismatch between the NestedGP predict_f variance shape and the GPflow convention",
    )

    # Check "good" match between aggregated model and gpr at training points
    np.testing.assert_array_almost_equal(
        mean_agg,
        mean_gpr,
        decimal=1,
        err_msg="mismatch between the NestedGP and GPR mean predictions",
    )
    np.testing.assert_array_almost_equal(
        var_agg,
        var_gpr,
        decimal=1,
        err_msg="mismatch between the NestedGP and GPR var predictions",
    )
