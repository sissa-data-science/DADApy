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

"""Module for testing the KStar class."""

import numpy as np
from dadapy import KStar

# define a basic dataset with 6 points
data = np.array([[0, 0], [0.1, 0], [0.2, 0], [4, 0], [4.1, 0], [4.2, 0]])


# TODO: Matteo, are these correct?
expected_kstar_low_Dthr = np.array([2, 2, 2, 2, 2, 2])
expected_kstar_high_Dthr = np.array([4, 4, 4, 4, 4, 4])


def test_compute_kstar_low_Dthr():
    """Test the compute_kstar method with low Dthr."""
    # create the KStar object
    kstar = KStar(coordinates=data)
    # compute kstar
    kstar.compute_kstar(Dthr=0.0)
    # check that the result is correct
    assert np.array_equal(kstar.kstar, expected_kstar_low_Dthr)


def test_compute_kstar_high_Dthr():
    """Test the compute_kstar method with high Dthr."""
    # create the KStar object
    kstar = KStar(coordinates=data)
    # compute kstar
    kstar.compute_kstar(Dthr=10000000.0)
    # check that the result is correct
    assert np.array_equal(kstar.kstar, expected_kstar_high_Dthr)


def test_set_kstar():
    """Test the set_kstar method."""
    # create the KStar object
    kstar = KStar(coordinates=data)
    # set kstar
    set_kstar = [1, 2, 3, 4, 5, 6]
    kstar.set_kstar(kstar=expected_kstar_low_Dthr)
    # check that the result is correct
    assert np.array_equal(kstar.kstar, set_kstar)
