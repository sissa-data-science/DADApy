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

"""Module for testing the Gride ID estimator."""

import os

import numpy as np
import pytest

from dadapy import IdEstimation


def test_compute_id_gride():
    """Test that the id estimations with gride work correctly."""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

    np.random.seed(0)

    X = np.load(filename)

    ie = IdEstimation(coordinates=X)

    # testing gride scaling
    ids, ids_err, rs = ie.return_id_scaling_gride()

    assert ids == pytest.approx(
        [2.00075528, 2.0321371, 1.91183327, 2.00055726, 1.6747242], abs=0.01
    )
    assert ids_err == pytest.approx(
        [0.20007553, 0.14664295, 0.09766772, 0.07223903, 0.04275823], abs=0.01
    )
    assert rs == pytest.approx(
        [0.39476585, 0.56740507, 0.80139545, 1.13457408, 1.64776878], abs=0.01
    )


def test_zero_dist():
    """Test that a warning message appear if there are overlapping datapoints."""
    X = np.array([[0, 0, 0], [0, 0, 0], [0.9, 0, 0]])
    de = IdEstimation(coordinates=X)

    with pytest.warns(UserWarning):
        de.return_id_scaling_gride()
