import numpy as np
import pytest

import gpflow

import guepard
from guepard.utilities import get_svgp_submodels, init_ssvgp_with_ensemble

np.random.seed(123456)


@pytest.mark.parametrize("num_latent", [1, 3])
def test_predict_f(num_latent):
    dim = 4
    num_data = 50
    num_split = 2
    np.random.seed(0)

    X = np.random.uniform(size=(num_data, dim))
    Y = np.cos(2 * np.pi * X[:, :num_latent]) + np.random.normal(
        0, np.sqrt(0.1), size=(num_data, 1)
    )
    Y = 1 * (Y > 0)  # this is a classif problem so we convert Y to binary values
    Xtest = np.random.uniform(size=(20, dim))

    Xl = np.array_split(X, num_split)  # in practice, kmeans clustering is recommended
    Yl = np.array_split(Y, num_split)

    kernel = gpflow.kernels.Matern12(lengthscales=0.2)
    gpflow.set_trainable(kernel, False)
    lik = gpflow.likelihoods.Bernoulli()
    gpflow.set_trainable(lik, False)

    # get submodels and create a sparse SVGP, we set maxiter > 0 so that the SVGPs distributions
    # do not match exactly the prior
    M = get_svgp_submodels(
        list(zip(Xl, Yl)), [3] * num_split, kernel, likelihood=lik, maxiter=2
    )
    Zs, q_mus, q_sqrts = init_ssvgp_with_ensemble(M)
    m_ssvgp = guepard.SparseSVGP(kernel, lik, Zs, q_mus, q_sqrts, whiten=False)

    # Compare with an SVGP initialised with the ensemble predictions at Z
    m_ens = guepard.EquivalentObsEnsemble(M)
    Z = m_ssvgp.inducing_variable.Z

    q_m, q_v = m_ens.predict_f(Z, full_cov=True)
    q_s = np.linalg.cholesky(q_v)
    m_svgp = gpflow.models.SVGP(
        inducing_variable=Z,
        likelihood=lik,
        kernel=kernel,
        q_mu=q_m,
        q_sqrt=q_s,
        whiten=False,
    )

    # Check shapes of output matches the GPflow convention
    mean_ssvgp, var_ssvgp = m_ssvgp.predict_f(Xtest, full_cov=False)
    mean_svgp, var_svgp = m_svgp.predict_f(Xtest, full_cov=False)

    np.testing.assert_array_almost_equal(
        mean_ssvgp.shape,
        mean_svgp.shape,
        err_msg="mismatch between the SSVGP predicted mean shape and the GPflow convention",
    )

    np.testing.assert_array_almost_equal(
        var_ssvgp.shape,
        var_svgp.shape,
        err_msg="mismatch between the SSVGP predicted variance shape and the GPflow convention",
    )

    # Check "good" match between aggregated model and gpr at training points
    mean_ssvgp, var_ssvgp = m_ssvgp.predict_f(Xtest, full_cov=True)
    mean_svgp, var_svgp = m_svgp.predict_f(Xtest, full_cov=True)

    np.testing.assert_array_almost_equal(
        mean_ssvgp,
        mean_svgp,
        decimal=3,
        err_msg="mismatch between the SSVGP and SVGP mean predictions",
    )
    np.testing.assert_array_almost_equal(
        var_ssvgp,
        var_svgp,
        decimal=4,
        err_msg="mismatch between the SSVGP and SVGP variance predictions",
    )
