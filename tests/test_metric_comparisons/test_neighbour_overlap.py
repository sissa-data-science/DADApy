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

"""Module for testing the neighbourhood overlap methods."""

import os

import numpy as np
import pytest

from dadapy import MetricComparisons

filename = os.path.join(os.path.split(__file__)[0], "../3d_gauss_small_z_var.npy")


def test_return_label_overlap():
    """Test that the label overlap works correctly."""
    X1 = np.load(filename)
    X2 = X1 + 1.0  # shifted gaussian

    X = np.vstack((X1, X2))  # datasets with two Gaussians

    # the labels distinguish the two Gaussians
    labels = np.ones(X.shape[0])
    labels[: X1.shape[0]] = 0

    mc = MetricComparisons(coordinates=X)
    mc.compute_distances()
    overlap = mc.return_label_overlap(labels=labels)

    assert overlap == pytest.approx(0.8676666666666668, 0.001)


def test_return_data_overlap():
    """Test that the label overlap works correctly."""
    X = np.load(filename)
    mc = MetricComparisons(coordinates=X)
    ov_data = mc.return_data_overlap(coordinates=X[:, :1], k=30, avg=True)
    assert pytest.approx(0.5833, 0.001) == ov_data
