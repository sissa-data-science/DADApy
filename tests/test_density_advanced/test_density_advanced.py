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

import os

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


expected_density_BMTI = np.array(
    [
        -1.698780556695925,
        -3.189031310462691,
        -2.523230763956809,
        -3.583852700030218,
        -1.7960043733709044,
        -3.0643213049156897,
        -3.977860544167944,
        -1.7921150971140554,
        -1.6797609470985766,
        -3.988603310090793,
        -2.188728113912356,
        -3.224052459516409,
        -3.721582556367274,
        -4.199926673166814,
        -1.658836709713782,
        -2.3769581305103724,
        -1.9130593343411615,
        -3.0670805680617694,
        -2.9879430640544475,
        -2.8543162418664916,
        -3.6531870983140053,
        -3.113710498689125,
        -2.192481141955024,
        -2.6462753213763226,
        -2.5283426346886047,
    ]
)


def test_density_BMTI():
    """Test the density_BMTI method."""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

    X = np.load(filename)[:25]

    da = DensityAdvanced(coordinates=X, maxk=10, verbose=True)
    da.compute_distances()
    da.set_id(2)
    da.compute_density_BMTI(alpha=0.99)

    assert np.allclose(da.log_den, expected_density_BMTI)
