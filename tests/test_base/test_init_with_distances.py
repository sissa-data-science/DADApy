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

"""Module for testing class initialisation with distance matrices."""

import numpy as np
from sklearn.metrics import pairwise_distances

from dadapy import Base


def test_distance_initialization():
    """Test initialization with distances."""
    X = np.array([[0, 0, 0], [0.5, 0, 0], [0.9, 0, 0]])

    dists = pairwise_distances(X)

    d = Base(distances=dists, maxk=1)

    expected_dists = np.array([[0.0, 0.5], [0.0, 0.4], [0.0, 0.4]])
    expected_indices = np.array([[0, 1], [1, 2], [2, 1]])

    assert (d.distances == expected_dists).all()
    assert (d.dist_indices == expected_indices).all()
