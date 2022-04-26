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

"""Module for testing information imbalance related methods."""

import os

import numpy as np
import pytest

from dadapy import MetricComparisons

filename = os.path.join(os.path.split(__file__)[0], "../3d_gauss_small_z_var.npy")


def test_information_imbalance_basics():
    """Test the information imbalance operations work correctly."""
    X = np.load(filename)

    mc = MetricComparisons(coordinates=X)

    mc.compute_distances()

    coord_list = [
        [
            0,
        ],
        [
            1,
        ],
        [
            2,
        ],
        [0, 1],
        [0, 2],
        [1, 2],
    ]

    imbalances = mc.return_inf_imb_full_selected_coords(coord_list)

    expected_imb = np.array(
        [
            [0.14400000000000002, 0.15, 0.968, 0.02, 0.1426, 0.1492],
            [0.5978, 0.5128, 0.9695999999999999, 0.02, 0.6208, 0.5434],
        ]
    )

    # Check we get the expected answer
    assert (imbalances == expected_imb).all()


def test_greedy_feature_selection_full():
    """Test that the information imbalance greedy optimisation works correctly."""
    expeted_coords = np.array([0, 1])
    expected_imbalances = np.array([[0.15, 0.51], [0.02, 0.02]])

    X = np.load(filename)

    mc = MetricComparisons(coordinates=X, maxk=X.shape[0] - 1)

    selected_coords, best_imbalances, all_imbalances = mc.greedy_feature_selection_full(
        n_coords=2
    )

    assert (selected_coords[-1] == expeted_coords).all()

    assert best_imbalances == pytest.approx(expected_imbalances, abs=0.01)


def test_return_inf_imb_matrix_of_coords():
    """Test inf imb calculation of all coordinates to all others."""
    X = np.load(filename)[:, [0, 1]]

    expected_matrix = np.array([[0.0, 1.02], [0.99, 0.0]])

    mc = MetricComparisons(coordinates=X, maxk=X.shape[0] - 1)

    matrix = mc.return_inf_imb_matrix_of_coords()

    assert matrix == pytest.approx(expected_matrix, abs=0.01)


def test_return_inf_imb_two_selected_coords():
    """Test information imbalance between selected coordinates."""
    X = np.load(filename)

    mc = MetricComparisons(coordinates=X, maxk=X.shape[0] - 1)

    imbalances = mc.return_inf_imb_two_selected_coords([0], [0, 1])

    assert imbalances == pytest.approx([0.598, 0.144], abs=0.001)
