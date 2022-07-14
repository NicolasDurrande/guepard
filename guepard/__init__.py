__version__ = "0.1.0"

from .gpr import GprPapl, get_gpr_submodels
from .sparse import SparseGuepard, get_svgp_submodels

__all__ = [
    "GprPapl",
    "get_gpr_submodels",
    "SparseGuepard",
    "get_svgp_submodels",
]
