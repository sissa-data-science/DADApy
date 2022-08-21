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

"""Module for testing the PAk density estimator."""

import os

import numpy as np

from dadapy import DensityEstimation

expected_log_den = [
    -1.26536091,
    -2.42577498,
    -1.59277014,
    -2.88670098,
    -1.58077996,
    -2.18950172,
    -3.74525393,
    -1.49297121,
    -1.21869391,
    -3.65211013,
    -1.65323983,
    -3.04360257,
    -3.40199876,
    -4.10092214,
    -1.15642398,
    -1.58343193,
    -1.21138508,
    -2.30480639,
    -2.84687316,
    -2.09987579,
    -3.42064175,
    -2.64021133,
    -2.00840896,
    -1.81506003,
    -1.53424819,
]


def test_compute_density_PAk():
    """Test that compute_density_PAk works correctly."""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

    X = np.load(filename)[:25]

    # optimized version
    de = DensityEstimation(coordinates=X)
    log_den_opt, log_den_err_opt = de.compute_density_PAk(optimized=True)

    # original version
    de = DensityEstimation(coordinates=X)
    log_den, log_den_err = de.compute_density_PAk(optimized=False)

    # np.set_printoptions(8)
    # print(de.log_den)

    # check consistency with ground truth
    assert np.allclose(log_den_opt, expected_log_den)

    # check consistency between original and optimized
    assert np.allclose(log_den, log_den_opt)

    assert np.allclose(log_den_err, log_den_err_opt)
