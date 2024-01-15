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

"""Module for testing differentiable information imbalance methods."""

import itertools as it

import numpy as np
import pytest

from dadapy import FeatureSelection
from dadapy.data import Data

rng = np.random.default_rng()


def test_optimise_imbalance_typing():
    data = rng.random(10, 5)

    for period in [
        [3],
        "faz",
        np.array([2, 2], dtype=np.int8),
        np.array([2, 2], dtype=np.float32),
    ]:
        feature_selection = FeatureSelection(data, period=period)
        pytest.raises(
            ValueError, feature_selection.optimize_kernel_imbalance(Data(data), 1)
        )

    for initial_gammas in [np.array([2, 2], np.float32), ["faz"]]:
        feature_selection = FeatureSelection(data)
        pytest.raises(
            ValueError,
            feature_selection.optimize_kernel_imbalance(
                Data(data), initial_gammas=initial_gammas
            ),
        )


def test_dist_matrix():
    for n_data in np.logspace(1, 2, 10, dtype=np.int16):
        for n_dim in np.logspace(1, 2, 10, dtype=np.int16):
            data = rng.random((n_data, n_dim))
            selector = FeatureSelection(data)
            for period, cythond in it.product(
                [None, np.ones(n_dim)], [False, True]
            ):  # TODO: should int, float also be accepted for period?
                dist_mat = selector.compute_dist_matrix(
                    data, period=period, cythond=cythond
                )
                assert dist_mat.shape[0] == n_data
                assert dist_mat.shape[1] == n_data


def test_rank_matrix():
    for n_data in np.logspace(1, 2, 10, dtype=np.int16):
        for n_dim in np.logspace(1, 2, 10, dtype=np.int16):
            # Construct data so that nearest point is always previous one
            data = np.zeros((n_data, n_dim), dtype=np.float16)
            data[:, :] = np.arange(n_data)[:, np.newaxis] ** 2

            for cythond in [False, True]:
                selector = FeatureSelection(data)
                ranks = selector.compute_rank_matrix(
                    data, period=None, cythond=cythond, distances=False
                )
                assert ranks.shape[0] == n_data
                assert ranks.shape[1] == n_data

                assert (
                    np.count_nonzero(np.diag(ranks, k=-1) == 1) == n_data - 1
                ), "Nearest neighbours not properly calculated."
