__version__ = "0.1.0"

from .equivalentobs import EquivalentObsEnsemble
from .utilities import get_gpr_submodels, get_svgp_submodels

__all__ = ["get_gpr_submodels", "get_svgp_submodels", "EquivalentObsEnsemble"]
