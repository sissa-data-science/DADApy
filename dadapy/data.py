# Copyright 2021-2023 The DADApy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
The *data* module contains the *Data* class.

Such a class inherits from all other classes defined in the package and as such it provides a convenient container of
all the algorithms implemented in Dadapy.
"""

import os

import numpy as np

from dadapy._utils.utils import cores
from dadapy.clustering import Clustering
from dadapy.density_advanced import DensityAdvanced
from dadapy.feature_weighting import FeatureWeighting
from dadapy.information_imbalance import InformationImbalance
from dadapy.neighborhood_overlap import NeighborhoodOverlap

np.set_printoptions(precision=2)
os.getcwd()


class Data(
    Clustering,
    DensityAdvanced,
    InformationImbalance,
    NeighborhoodOverlap,
    FeatureWeighting,
):
    """Data class."""

    def __init__(
        self,
        coordinates=None,
        distances=None,
        maxk=None,
        period=None,
        verbose=False,
        n_jobs=cores,
        working_memory=1024,
        rng_seed=42,
    ):
        """Initialise a Data object, container of all DADApy methods.

        It is initialised with a set of coordinates or a set of
        distances, and all methods can be called on the generated class instance.

        Args:
            coordinates (np.ndarray(float)): the data points loaded, of shape (N , dimension of embedding space)
            distances (np.ndarray(float)): A matrix of dimension N x mask containing distances between points
            maxk (int): maximum number of neighbours to be considered for the calculation of distances
            verbose (bool): whether you want the code to speak or shut up
            n_jobs (int): number of cores to be used
            working_memory (int): working memory (TODO: currently unused)
            rng_seed (int): seed used to build ``self.rng``. Setting this makes every randomised
                method called on the instance reproducible.
        """
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            period=period,
            verbose=verbose,
            n_jobs=n_jobs,
            rng_seed=rng_seed,
        )
