import gpflow
import numpy as np
import pytest

from guepard import GprPapl, get_gpr_submodels

np.random.seed(123456)


def test_get_gpr_submodels():
    data = [
        (np.random.uniform(size=(10, 2)), np.random.normal(size=(10, 1)))
        for _ in range(3)
    ]
    kernel = gpflow.kernels.Matern32()
    M = get_gpr_submodels(data, kernel)

    assert (
        len(M) == 3
    ), "The length of the model list isn't equal to the length of\
        the data list"

    # smoke test on model prediction
    M[1].predict_f(np.random.uniform(np.random.uniform(size=(3, 2))))


@pytest.mark.parametrize("num_latent", [1, 2])
def test_papl_predict_f_marginals(num_latent):
    dim = 3
    num_data = 200
    num_split = 3
    X = np.random.uniform(size=(num_data, dim))
    Y = np.cos(X[:, :num_latent]) + np.random.normal(
        0, np.sqrt(0.1), size=(num_data, 1)
    )

    Xl = np.array_split(X, num_split)
    Yl = np.array_split(Y, num_split)

    kernel = gpflow.kernels.Matern32()

    # make submodels and aggregate them
    M = get_gpr_submodels(list(zip(Xl, Yl)), kernel)
    m_agg = GprPapl(M)

    # make a GPR model as baseline
    m_gpr = gpflow.models.GPR((X, Y), kernel, noise_variance=0.1)

    # Check "good" match between aggregated model and gpr at training points
    mean_agg, var_agg = m_agg.predict_f_marginals(X)
    mean_gpr, var_gpr = m_gpr.predict_f(X)

    np.testing.assert_array_almost_equal(
        mean_agg,
        mean_gpr,
        decimal=1,
        err_msg="mismatch between the PALP and GPR mean predictions",
    )
    np.testing.assert_array_almost_equal(
        var_agg,
        var_gpr,
        decimal=1,
        err_msg="mismatch between the PALP and GPR var predictions",
    )

    # Check "good" match between aggregated model and prior far away from training data
    mean_agg, var_agg = m_agg.predict_f_marginals(X + 10.0)

    np.testing.assert_array_almost_equal(
        mean_agg,
        np.zeros_like(mean_agg),
        decimal=4,
        err_msg="PALP mean does not revert to prior far away from training data",
    )
    np.testing.assert_array_almost_equal(
        var_agg,
        np.ones_like(var_agg),
        decimal=4,
        err_msg="PALP variance does not revert to prior far away from training data",
    )


@pytest.mark.parametrize("num_latent", [1, 2])
def test_papl_predict_f(num_latent):
    dim = 3
    num_data = 200
    num_split = 3
    X = np.random.uniform(size=(num_data, dim))
    Y = np.cos(X[:, :num_latent]) + np.random.normal(
        0, np.sqrt(0.1), size=(num_data, 1)
    )

    Xl = np.array_split(X, num_split)
    Yl = np.array_split(Y, num_split)

    kernel = gpflow.kernels.Matern32()

    # make submodels and aggregate them
    M = get_gpr_submodels(list(zip(Xl, Yl)), kernel)
    m_agg = GprPapl(M)

    # make a GPR model as baseline
    m_gpr = gpflow.models.GPR((X, Y), kernel, noise_variance=0.1)

    # Check "good" match between aggregated model and gpr at training points
    mean_agg, var_agg = m_agg.predict_f(X, full_cov=True)
    mean_gpr, var_gpr = m_gpr.predict_f(X, full_cov=True)

    np.testing.assert_array_almost_equal(
        mean_agg,
        mean_gpr,
        decimal=4,
        err_msg="mismatch between the PALP and GPR mean predictions",
    )
    np.testing.assert_array_almost_equal(
        var_agg,
        var_gpr,
        decimal=4,
        err_msg="mismatch between the PALP and GPR var predictions",
    )

    # Check "good" match between aggregated model and prior far away from training data
    mean_agg, var_agg = m_agg.predict_f(X + 10.0)

    np.testing.assert_array_almost_equal(
        mean_agg,
        np.zeros_like(mean_agg),
        decimal=5,
        err_msg="PALP mean does not revert to prior far away from training data",
    )
    np.testing.assert_array_almost_equal(
        var_agg[0, :, :],
        m_gpr.kernel.K(X + 10.0),
        decimal=5,
        err_msg="PALP variance does not revert to prior far away from training data",
    )
