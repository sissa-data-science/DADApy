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

"""Module for testing the DensityAdvanced class."""

import numpy as np

from dadapy import DensityAdvanced

# define a basic dataset with 6 points
data = np.array([[0, 0], [0.15, 0], [0.2, 0], [4, 0], [4.1, 0], [4.2, 0]])


def test_compute_grads():
    """Test the compute_grads method."""
    # define the expected gradient
    expected_grads = np.array(
        [
            [13.124999999999995, 0],
            [-6.66666666666666, 0.0],
            [-9.374999999999998, 0.0],
            [11.249999999999973, 0.0],
            [1.3322676295501972e-13, 0.0],
            [-11.250000000000007, 0.0],
        ]
    )

    expected_grad_vars = [
        [2.6367187500000426, 0.0],
        [133.33333333333337, 0.0],
        [23.73046874999999, 0.0],
        [10.546875000000107, 0.0],
        [675.0000000000107, 0.0],
        [10.546874999999883, 0.0],
    ]

    da = DensityAdvanced(coordinates=data, maxk=3, verbose=True)
    da.compute_distances()
    da.set_id(1)
    da.set_kstar(3)
    da.compute_grads(comp_covmat=False)

    assert np.allclose(da.grads, expected_grads)
    assert np.allclose(da.grads_var, expected_grad_vars)


def test_compute_deltaFs():
    """Test the compute_deltaFs method."""
    # define the expected deltaFs
    expected_Fij_array = np.array(
        [5.999999999999997, 0.0, 0.0, 0.0, 0.0, 3.0000000000000133]
    )
    expected_Fij_var_array = np.array([1.2789769243681803e-15, 0.0, 0.0, 0.0, 0.0, 0.0])

    da = DensityAdvanced(coordinates=data, maxk=2, verbose=True)
    da.compute_distances()
    da.set_id(1)
    da.set_kstar(2)
    da.compute_deltaFs()
    assert np.allclose(da.Fij_array, expected_Fij_array)
    assert np.allclose(da.Fij_var_array, expected_Fij_var_array)


def test_density_BMTI():
    """Test the density_BMTI method."""
    # define the expected density
    expected_density = np.array(
        [
            -0.06290097151904936,
            -0.023556982206034104,
            -0.011088481060296614,
            -0.004762206359420831,
            -0.02968801181576885,
            -0.05593299286955579,
        ]
    )

    da = DensityAdvanced(coordinates=data, maxk=3, verbose=True)
    da.compute_distances()
    da.set_id(1)
    da.set_kstar(4)
    da.compute_density_BMTI()
    assert np.allclose(da.log_den, expected_density)
