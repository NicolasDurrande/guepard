import gpflow


from gpflow.base import RegressionData
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from typing import List, Optional


def get_gpr_submodels(data_list: List[RegressionData],
                      kernel: Kernel,
                      mean_function: Optional[MeanFunction] = None,
                      noise_variance: float = .1):
    """
    Helper function to build a list of GPflow GPR submodels from a list of datasets and a GP prior, and a likelihood variance
    """
    return [gpflow.models.GPR(data, kernel, mean_function, noise_variance) for data in data_list]


