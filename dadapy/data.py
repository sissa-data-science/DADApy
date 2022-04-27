# Copyright 2021-2022 The DADApy Authors. All Rights Reserved.
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

import multiprocessing
import os

import numpy as np

from dadapy.clustering import Clustering
from dadapy.metric_comparisons import MetricComparisons

cores = multiprocessing.cpu_count()
np.set_printoptions(precision=2)
os.getcwd()


class Data(Clustering, MetricComparisons):
    """This class is a container of all the method of DADApy. It is initialised with a set of coordinates or a set of
    distances, and all methods can be called on the generated class instance.

    Attributes:

    """

    def __init__(
        self,
        coordinates=None,
        distances=None,
        maxk=None,
        verbose=False,
        njobs=cores,
        working_memory=1024,
    ):

        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            verbose=verbose,
            njobs=njobs,
        )
