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

"""Module for testing the 2NN ID estimator."""

import math
import os

import numpy as np
import pytest

from dadapy import IdEstimation

filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

X = np.load(filename)


def test_compute_id_2nn():
    """Test that the id estimations with 2NN work correctly."""
    ie = IdEstimation(coordinates=X, rng_seed=42)

    ie.compute_id_2NN()
    assert ie.intrinsic_dim == pytest.approx(1.85, abs=0.01)

    # testing 2NN scaling
    ids, ids_err, rs = ie.return_id_scaling_2NN(return_sizes=False)

    assert ids == pytest.approx([1.85491, 2.03909, 2.28923, 2.41457], abs=0.01)
    assert ids_err == pytest.approx([0.0, 0.04846, 0.39487, 0.20226], abs=0.01)
    assert rs == pytest.approx([0.39476, 0.52098, 0.73865, 1.15109], abs=0.01)


def test_compute_id_2nn_wprior():
    """Test that the id estimation with a Bayesian 2NN works correctly."""
    ie = IdEstimation(coordinates=X)

    ie.compute_id_2NN_wprior()

    assert ie.intrinsic_dim == pytest.approx(1.8722, abs=0.0001)
