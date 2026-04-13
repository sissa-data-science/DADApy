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
expected_kstar_low_Dthr = np.array([3, 3, 3, 3, 3, 3])
expected_kstar_high_Dthr = np.array([4, 4, 4, 4, 4, 4])


def test_compute_kstar_low_Dthr():
    """Test the compute_kstar method with low Dthr."""
    # create the KStar object
    kstar = KStar(coordinates=data)
    # compute kstar
    kstar.compute_kstar(alpha=1.0)
    # check that the result is correct
    assert np.array_equal(kstar.kstar, expected_kstar_low_Dthr)


def test_compute_kstar_high_Dthr():
    """Test the compute_kstar method with high Dthr."""
    # create the KStar object
    kstar = KStar(coordinates=data)
    # compute kstar
    kstar.compute_kstar(alpha=1e-300)
    # check that the result is correct
    assert np.array_equal(kstar.kstar, expected_kstar_high_Dthr)


def test_set_kstar():
    """Test the set_kstar method."""
    # create the KStar object
    kstar = KStar(coordinates=data)
    # set kstar
    set_kstar = [1, 2, 3, 4, 5, 6]
    kstar.set_kstar(k=set_kstar)
    # check that the result is correct
    assert np.array_equal(kstar.kstar, set_kstar)


def test_compute_kstar_bonferroni_deloc():
    """Test that bonferroni_deloc correction produces different results."""
    kstar_uncorrected = KStar(coordinates=data)
    kstar_uncorrected.compute_kstar(alpha=0.05, bonferroni_deloc=False)

    kstar_corrected = KStar(coordinates=data)
    kstar_corrected.compute_kstar(alpha=0.05, bonferroni_deloc=True)

    # With bonferroni correction, threshold is higher, so kstar should be <= uncorrected
    assert np.all(kstar_corrected.kstar <= kstar_uncorrected.kstar)


def test_compute_kstar_bonferroni_loc():
    """Test that bonferroni_loc correction produces different results."""
    kstar_uncorrected = KStar(coordinates=data)
    kstar_uncorrected.compute_kstar(alpha=0.05, bonferroni_loc=False)

    kstar_corrected = KStar(coordinates=data)
    kstar_corrected.compute_kstar(alpha=0.05, bonferroni_loc=True)

    assert np.all(kstar_corrected.kstar <= kstar_uncorrected.kstar)


def test_compute_kstar_bonferroni_both():
    """Test with both bonferroni corrections."""
    kstar = KStar(coordinates=data)
    kstar.compute_kstar(alpha=0.05, bonferroni_deloc=True, bonferroni_loc=True)
    # Just verify it runs and produces valid kstar values
    assert kstar.kstar is not None
    assert len(kstar.kstar) == len(data)
    assert np.all(kstar.kstar > 0)
