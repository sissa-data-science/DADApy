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

"""Module for testing the Binomial ID estimator."""


import os

import numpy as np
import pytest

from dadapy import IdEstimation

filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

X = np.load(filename)


def test_compute_id_2NN():
    """Test that the id estimations with the binomial estimator works correctly."""
    np.random.seed(0)

    ie = IdEstimation(coordinates=X)

    id_b = ie.compute_id_binomial_rk(0.2, 0.5)
    assert id_b == pytest.approx([2.08426, 0.33112, 0.150000], abs=1e-4, rel=1e-2)

    id_b = ie.compute_id_binomial_k(5, 0.5)
    assert id_b == pytest.approx([1.98391, 0.123781, 0.56159], abs=1e-4, rel=1e-2)
