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

"""Module for testing removal of identical points method."""

import numpy as np
import pytest

from dadapy import Base


def test_identical_points():
    """Test the removal of identical points."""
    X = np.array([[2, 1, 3], [1, 2, 3], [1, 2, 3]])

    d = Base(X, verbose=True)

    d.compute_distances()

    d.remove_identical_points()

    expected_X = np.array([[1, 2, 3], [2, 1, 3]])
    expected_dist_indices = np.array([[0, 1], [1, 0]])
    expected_dists = np.array([[0.0, 1.4142135623730951], [0.0, 1.4142135623730951]])

    assert pytest.approx(d.X) == expected_X
    assert pytest.approx(d.dist_indices) == expected_dist_indices
    assert pytest.approx(d.distances) == expected_dists
