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

"""Module for testing the density interpolators."""

import os

import numpy as np
import pytest

from dadapy import DensityEstimation


def test_density_estimation_kNN():
    """Test the kNN interpolator is coherent with the kNN estimator."""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

    X = np.load(filename)[:25]

    k = 5

    de = DensityEstimation(coordinates=X)

    computed, _ = de.compute_density_kNN(k)

    interpolated, _ = de.return_interpolated_density_kNN(X, k)

    diff = computed - interpolated

    expected_diff = np.array([np.log(k) - np.log(k - 1)] * len(diff))

    assert diff == pytest.approx(expected_diff, abs=1e-6)


def test_interpolated_density_kNN_optionally_returns_kstar():
    """Test that the kNN interpolator optionally returns its fixed kstar."""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")
    X = np.load(filename)[:25]
    k = 5

    de = DensityEstimation(coordinates=X)

    result = de.return_interpolated_density_kNN(X, k)
    result_with_kstar = de.return_interpolated_density_kNN(X, k, return_kstar=True)

    assert len(result) == 2
    for actual, expected in zip(result_with_kstar[:2], result):
        assert actual == pytest.approx(expected)
    assert np.array_equal(result_with_kstar[2], np.full(X.shape[0], k))


@pytest.mark.parametrize(
    "interpolator_name",
    ["return_interpolated_density_kstarNN", "return_interpolated_density_PAk"],
)
def test_adaptive_interpolated_density_optionally_returns_kstar(interpolator_name):
    """Test that adaptive interpolators optionally return their optimal kstar."""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")
    X = np.load(filename)[:25]

    de = DensityEstimation(coordinates=X)
    interpolator = getattr(de, interpolator_name)

    result = interpolator(X)
    result_with_kstar = interpolator(X, return_kstar=True)

    assert len(result) == 2
    for actual, expected in zip(result_with_kstar[:2], result):
        assert actual == pytest.approx(expected)
    assert result_with_kstar[2].shape == (X.shape[0],)
    assert np.issubdtype(result_with_kstar[2].dtype, np.integer)
