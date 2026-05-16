"""DADApy: a Python package for Distance-based Analysis of DAta-manifolds."""

import sys

from ._utils.utils import *  # noqa: F401,F403
from .base import Base  # noqa: F401
from .clustering import Clustering  # noqa: F401
from .data import Data  # noqa: F401
from .data_sets import DataSets  # noqa: F401
from .density_advanced import DensityAdvanced  # noqa: F401
from .density_estimation import DensityEstimation  # noqa: F401
from .feature_weighting import FeatureWeighting  # noqa: F401
from .id_discrete import IdDiscrete  # noqa: F401
from .id_estimation import IdEstimation  # noqa: F401
from .kstar import KStar  # noqa: F401
from .metric_comparisons import MetricComparisons  # noqa: F401
from .neigh_graph import NeighGraph  # noqa: F401

if sys.version_info >= (3, 9):
    from .causal_graph import CausalGraph  # noqa: F401
    from .diff_imbalance import DiffImbalance  # noqa: F401
    from .hamming import BID, Hamming  # noqa: F401
