__version__ = "0.1.0"

from .baselines import Ensemble, EnsembleMethods, GPEnsemble, NestedGP, WeightingMethods
from .equivalentobs import EquivalentObsEnsemble
from .sparseSVGP import SparseSVGP
from .utilities import get_gpr_submodels, get_svgp_submodels, init_ssvgp_with_ensemble

__all__ = [
    "get_gpr_submodels",
    "get_svgp_submodels",
    "EquivalentObsEnsemble",
    "Ensemble",
    "GPEnsemble",
    "NestedGP",
    "EnsembleMethods",
    "WeightingMethods",
    "SparseSVGP",
    "init_ssvgp_with_ensemble",
]
