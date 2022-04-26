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

"""Module for testing the kpeaks density estimator."""

import os

import numpy as np

from dadapy import DensityEstimation

expected_log_den = np.array([23.0] * 25)
expected_log_den[13] = 16.0


def test_compute_density_kpeaks():
    """Test that compute_density_kpeaks works correctly."""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

    X = np.load(filename)[:25]

    de = DensityEstimation(coordinates=X)

    de.compute_density_kpeaks()

    assert (de.log_den == expected_log_den).all()
