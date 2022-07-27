__version__ = "0.1.0"

from .utilities import get_gpr_submodels, get_svgp_submodels
from .equivalentobs import EquivalentObsEnsemble

__all__ = [
    "get_gpr_submodels",
    "get_svgp_submodels",
    "EquivalentObsEnsemble"
]
