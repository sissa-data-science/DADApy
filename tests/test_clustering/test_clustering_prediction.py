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

"""Module for testing DP clustering."""

import os

import numpy as np

from dadapy import Clustering

filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

X = np.load(filename)

expected_cluster_assignment = np.array(
    [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
)


def test_predict_DP_PAk():
    """Test the prediction using PAk clustering works correctly."""
    cl = Clustering(coordinates=X[25:-25])
    cl.compute_clustering_ADP(halo=False)
    preds0, _ = cl.predict_cluster_DP(X[-25:], density_est="PAk")
    preds1, _ = cl.predict_cluster_DP(X[:25], density_est="PAk")
    assert (preds1 == cl.cluster_assignment[:25]).all()
    assert (preds0 == cl.cluster_assignment[-25:]).all()


def test_predict_DP_kstarNN():
    """Test the prediction using kstarNN clustering works correctly."""
    cl = Clustering(coordinates=X[25:-25])
    cl.compute_clustering_ADP(halo=False)
    preds0, _ = cl.predict_cluster_DP(X[-25:], density_est="kstarNN")
    preds1, _ = cl.predict_cluster_DP(X[:25], density_est="kstarNN")
    assert (preds1 == cl.cluster_assignment[:25]).all()
    assert (preds0 == cl.cluster_assignment[-25:]).all()


def test_predict_smooth():
    """Test the smooth prediction using kstar neighbour distances works correctly."""
    cl = Clustering(coordinates=X[25:-25])
    cl.compute_clustering_ADP(halo=False)
    preds0, _ = cl.predict_cluster_inverse_distance_smooth(X[-25:])
    preds1, _ = cl.predict_cluster_inverse_distance_smooth(X[:25])
    assert (preds1 == cl.cluster_assignment[:25]).all()
    assert (preds0 == cl.cluster_assignment[-25:]).all()
