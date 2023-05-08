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

import os

import numpy as np
import pytest

from dadapy import DensityEstimation

filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

X = np.load(filename)


def test_compute_id_2NN():
    """Test that the id estimations with 2NN work correctly."""
    np.random.seed(0)

    de = DensityEstimation(coordinates=X)

    de.compute_id_2NN()
    assert pytest.approx(de.intrinsic_dim, abs=0.01) == 1.85

    # testing 2NN scaling
    ids, ids_err, rs = de.return_id_scaling_2NN()

    assert ids == pytest.approx([1.85, 1.77, 1.78, 2.31], abs=0.01)
    assert ids_err == pytest.approx([0.0, 0.11, 0.15, 0.19], abs=0.01)
    assert rs == pytest.approx([0.39, 0.58, 0.78, 1.13], abs=0.01)


def test_compute_id_2NN_wprior():
    """Test that the id estimation with a Bayesian 2NN works correctly."""
    de = DensityEstimation(coordinates=X)

    de.compute_id_2NN_wprior()

    assert pytest.approx(de.intrinsic_dim, abs=0.0001) == 1.8722
