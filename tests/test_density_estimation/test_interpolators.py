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

"""Module for testing the density interpolators."""

import os

import numpy as np

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

    assert (diff == expected_diff).all()
