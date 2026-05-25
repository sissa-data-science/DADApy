"""DADApy: a Python package for Distance-based Analysis of DAta-manifolds."""

import sys

from ._utils.utils import *  # noqa: F401,F403
from .base import Base
from .clustering import Clustering
from .data import Data
from .data_sets import DataSets
from .density_advanced import DensityAdvanced
from .density_estimation import DensityEstimation
from .feature_weighting import FeatureWeighting
from .id_discrete import IdDiscrete
from .id_estimation import IdEstimation
from .kstar import KStar
from .metric_comparisons import MetricComparisons
from .neigh_graph import NeighGraph

__all__ = [
    "Base",
    "Clustering",
    "Data",
    "DataSets",
    "DensityAdvanced",
    "DensityEstimation",
    "FeatureWeighting",
    "IdDiscrete",
    "IdEstimation",
    "KStar",
    "MetricComparisons",
    "NeighGraph",
]

try:
    from .causal_graph import CausalGraph
    from .diff_imbalance import DiffImbalance
    from .hamming import BID, Hamming

    __all__ += ["CausalGraph", "DiffImbalance", "BID", "Hamming"]
except (ImportError, RuntimeError):
    # JAX-dependent classes unavailable (e.g., this exception is raised in 
    # joblib worker subprocesses where GPU context cannot be re-initialized 
    # from the parent process).
    pass
