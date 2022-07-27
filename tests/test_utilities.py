from dataclasses import dataclass

import numpy as np
import pytest

import gpflow

from guepard.utilities import get_gpr_submodels, get_svgp_submodels


@dataclass(frozen=True)
class CONSTS:
    noise_var = 0.1
    num_splits = 3
    num_inducing = 2


@pytest.fixture
def data():
    def f(x):
        return np.sin(10 * x[:, :1]) + 3.0 * x[:, :1]

    X = np.linspace(0, 1, 101)[:, None]
    Y = f(X) + np.sqrt(CONSTS.noise_var) * np.random.normal(size=X.shape)
    return X, Y


def test_get_gpr_submodels():
    data = [
        (np.random.uniform(size=(10, 2)), np.random.normal(size=(10, 1)))
        for _ in range(3)
    ]
    kernel = gpflow.kernels.Matern32()
    M = get_gpr_submodels(data, kernel)

    assert (
        len(M) == 3
    ), "The length of the model list isn't equal to the length of the data list"

    # smoke test on model prediction
    M[1].predict_f(np.random.uniform(np.random.uniform(size=(3, 2))))


def test_get_svgp_submodels(data):
    X, Y = data
    ns = CONSTS.num_splits
    x_list = np.array_split(X, ns)  # list of num_split np.array
    y_list = np.array_split(Y, ns)
    kernel = gpflow.kernels.Matern32()
    num_inducing_list = [CONSTS.num_inducing] * ns
    data_list = list(zip(x_list, y_list))
    models = get_svgp_submodels(
        data_list,
        num_inducing_list,
        kernel,
        noise_variance=CONSTS.noise_var,
        maxiter=-1,
    )

    assert (
        len(models) == ns
    ), "The length of the model list isn't equal to the length of the data list"

    # smoke test on model prediction
    models[1].predict_f(np.random.uniform(np.random.uniform(size=(3, 1))))
