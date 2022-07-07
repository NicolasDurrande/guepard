from dataclasses import dataclass

import numpy as np
import pytest

import gpflow

from guepard.sparse import SparsePapl, get_svgp_submodels


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


@pytest.fixture(name="submodels")
def _fixture_submodels(data):
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
    return models


def test_sparsepapl(submodels):
    m = SparsePapl(submodels)
    ensemble = m.get_ensemble_svgp()
    assert len(ensemble.inducing_variable) == (CONSTS.num_splits * CONSTS.num_inducing)
