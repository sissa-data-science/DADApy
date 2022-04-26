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

"""Module for testing the kstarNN density estimator."""

import os

import numpy as np

from dadapy import DensityEstimation

expected_log_den = [
    -3.13188653,
    -3.8628515,
    -3.87037581,
    -4.3656842,
    -3.25613168,
    -3.91041562,
    -4.42116092,
    -3.0793219,
    -3.14011977,
    -4.33251233,
    -3.2642416,
    -3.76676413,
    -4.64696587,
    -4.27332111,
    -3.1458499,
    -3.67534274,
    -3.15572509,
    -4.28226485,
    -3.53777735,
    -4.10666217,
    -3.8753995,
    -4.37380939,
    -3.5579838,
    -3.98474186,
    -3.77986474,
]


def test_compute_density_kstarNN():
    """Test that compute_density_kstarNN works correctly."""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

    X = np.load(filename)[:25]

    de = DensityEstimation(coordinates=X)

    de.compute_density_kstarNN()

    assert np.allclose(de.log_den, expected_log_den)
