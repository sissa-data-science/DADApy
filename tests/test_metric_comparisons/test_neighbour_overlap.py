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


def test_return_overlap_coords():
    """Test that the neighbourhood overlap works correctly."""
    X = np.load(filename)

    mc = MetricComparisons(coordinates=X)
    mc.compute_distances()
    # check equivalence of x-y subspace with the full x-y-z space on this dataset
    overlap = mc.return_overlap_coords(coords1=[0, 1, 2], coords2=[0, 1])

    assert overlap == pytest.approx(0.992, abs=0.0001)


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

    assert overlap == 0.8676666666666668


def test_return_label_overlap_selected_coords():
    """Test that the label overlap works correctly."""
    X = np.load(filename)

    # labels simply distinguish left from right
    labels = np.ones(X.shape[0])
    labels[np.where(X[:, 0] < 0)] = 0

    coord_list = [[0, 1, 2], [0], [1], [2]]
    # coordinate 0 is expected to better distinguish left and right
    expected_overlaps = [0.7783333333333333, 0.927, 0.494, 0.48533333333333334]

    mc = MetricComparisons(coordinates=X, maxk=X.shape[0] - 1)

    mc.compute_distances()

    overlaps = mc.return_label_overlap_selected_coords(
        labels=labels, coord_list=coord_list
    )

    assert overlaps == pytest.approx(expected_overlaps)


def test_return_data_overlap():
    """Test that the label overlap works correctly."""
    X = np.load(filename)
    mc = MetricComparisons(coordinates=X)
    ov_data = mc.return_data_overlap(coordinates=X[:, :1], k=30, avg=True)
    assert pytest.approx(0.58333) == ov_data
