from papl import __version__
import numpy as np
from papl.GPR_submodels import get_gpr_submodels

import gpflow
def test_version():
    assert __version__ == "0.1.0"

def test_get_gpr_submodels():
    data = ((np.random.uniform(size=(10, 2)), np.random.normal(size=(10, 1))) for _ in range(3))
    kernel = gpflow.kernels.Matern32()
    M = get_gpr_submodels(data, kernel)

    assert len(M)==3, "The length of the model list isn't equal to the length of\
        the data list"
    
    # smoke test on model prediction
    M[1].predict_f(np.random.uniform(np.random.uniform(size=(3, 2))))
