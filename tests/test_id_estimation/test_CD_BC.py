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

"""Module for testing the correlaion dimension and box counting ID estimators."""


import os

import numpy as np
import pytest

from dadapy import IdEstimation
from dadapy._utils.id_estimation import box_counting as BC
from dadapy._utils.id_estimation import correlation_integral as CD

filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

X = np.load(filename)


def test_compute_id_CD_BC():
    """Test that the id estimations with the binomial estimator works correctly."""
    np.random.seed(0)

    ie = IdEstimation(coordinates=X, maxk=len(X) - 1)
    ie.compute_distances()

    id_cd, _, _ = CD(ie.distances, np.linspace(0.25, 1.75, 10), plot=False)
    assert id_cd == pytest.approx(
        [
            1.88925,
            1.87676,
            1.91440,
            1.90752,
            1.89357,
            1.87727,
            1.854341,
            1.83551,
            1.81872,
        ],
        abs=1e-4,
        rel=1e-2,
    )

    id_bc, _ = BC(
        X, (-5, 5), np.linspace(1.0, 3, 10), n_offsets=10, plot=False, verb=False
    )
    assert id_bc == pytest.approx(
        [1.03, 1.27, 1.37, 1.27, 1.31, 1.36, 1.37], abs=1e-2, rel=1e-2
    )
