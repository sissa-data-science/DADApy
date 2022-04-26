# Copyright 2021 The DADApy Authors. All Rights Reserved.
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

"""Module for testing the distance computations."""

import numpy as np
import pytest

from dadapy import Base


def test_compute_distances():
    """Test the computation of distances."""
    X = np.array([[0, 0, 0], [0.5, 0, 0], [0.9, 0, 0]])

    base = Base(coordinates=X)

    base.compute_distances()

    expected_dists = [[0.0, 0.5, 0.9], [0.0, 0.4, 0.5], [0.0, 0.4, 0.9]]

    expected_ind = [[0, 1, 2], [1, 2, 0], [2, 1, 0]]

    assert pytest.approx(base.distances) == expected_dists
    assert pytest.approx(base.dist_indices) == expected_ind

    base.compute_distances(period=1.1, metric="manhattan")

    expected_dists = [[0.0, 0.2, 0.5], [0.0, 0.4, 0.5], [0.0, 0.2, 0.4]]

    expected_ind = [[0, 2, 1], [1, 2, 0], [2, 0, 1]]

    assert pytest.approx(base.distances) == expected_dists
    assert pytest.approx(base.dist_indices) == expected_ind
