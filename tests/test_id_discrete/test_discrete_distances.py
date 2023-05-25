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

"""Module for testing the I3D estimator."""

import numpy as np

from dadapy import IdDiscrete
from dadapy._utils import discrete_functions as df


def test_hamming_condensed():
    """Test the discrete id estimator with canonical distances storing."""
    N = 100
    colors = 4
    d = 20
    rng = np.random.default_rng(12345)

    X = rng.integers(0, colors, size=(N, d))

    I3D = IdDiscrete(X)
    I3D.compute_distances(metric="hamming", condensed=True, d_max=20)

    dist, _ = df.hamming_distances_idx(X, d_max=20, maxk_ind=20)

    print(I3D.distances[0], dist[0])
    diff = sum([sum(dist[i] - I3D.distances[i]) for i in range(len(X))])
    assert diff == 0


def test_manhattan_condensed():
    """Test the discrete id estimator with cumulative distances storing."""
    N = 100
    box = 20
    d = 5
    rng = np.random.default_rng(12345)

    X = rng.integers(0, box, size=(N, d))

    # use PBC
    I3D = IdDiscrete(X, condensed=True)
    I3D.compute_distances(metric="manhattan", period=box, d_max=50)

    dist, _ = df.manhattan_distances_idx(X, d_max=50, maxk_ind=20, period=box)
    diff = sum([sum(dist[i] - I3D.distances[i]) for i in range(len(X))])
    assert diff == 0

    # without PBC
    I3D = IdDiscrete(X, condensed=True)
    I3D.compute_distances(metric="manhattan", d_max=80)

    dist, ind = df.manhattan_distances_idx(X, d_max=80, maxk_ind=20)

    diff = sum([sum(dist[i] - I3D.distances[i]) for i in range(len(X))])
    assert diff == 0
