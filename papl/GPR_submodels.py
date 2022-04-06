import gpflow


from gpflow.base import RegressionData
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction


def get_gpr_submodels(data_list: RegressionData,
                      kernel: Kernel,
                      mean_function: Optional[MeanFunction] = None,
                      noise_variance: float = 1.0):
    """
    Helper function to build a list of GPflow GPR submodels from a list of dataset and some prior parameters
    """
    return [gpflow.models.GPR(data, kernel, mean_function, noise_variance) for data in data_list]


