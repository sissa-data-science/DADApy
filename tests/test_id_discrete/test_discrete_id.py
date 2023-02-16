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

"""Module for testing the I3D estimator."""

import numpy as np
import pytest

from dadapy.id_discrete import IdDiscrete


def test_distances():
    """Test the discrete id estimator with canonical distances storing."""
    N = 500
    box = 20
    d = 5
    rng = np.random.default_rng(12345)

    X = rng.integers(0, box, size=(N, d))

    I3D = IdDiscrete(X, maxk=X.shape[0])
    I3D.compute_distances(metric="manhattan", period=box, condensed=False)

    I3D.compute_id_binomial_k(k=25, shell=False, ratio=0.5)
    assert I3D.intrinsic_dim == pytest.approx(5.018707133975087)

    I3D.compute_id_binomial_k(k=4, shell=True, ratio=0.5)
    assert I3D.intrinsic_dim == pytest.approx(5.602713972478171)

    I3D.compute_id_binomial_lk(
        lk=10, ln=5, method="bayes", plot=False, subset=np.arange(100)
    )
    assert I3D.intrinsic_dim == pytest.approx(4.906996203360682)

    ks, pv = I3D.model_validation_full(cdf=False)
    assert pv > 0.005


def test_distances_condensed():
    """Test the discrete id estimator with cumulative distances storing."""
    N = 500
    box = 20
    d = 5
    rng = np.random.default_rng(12345)

    X = rng.integers(0, box, size=(N, d))

    I3Dc = IdDiscrete(X, condensed=True)
    I3Dc.compute_distances(metric="manhattan", period=box, d_max=d * box)

    I3Dc.compute_id_binomial_k(k=25, shell=False, ratio=0.5)
    assert I3Dc.intrinsic_dim == pytest.approx(5.018707133975087)

    I3Dc.compute_id_binomial_k(k=4, shell=True, ratio=0.5)
    assert I3Dc.intrinsic_dim == pytest.approx(5.602713972478171)

    I3Dc.compute_id_binomial_lk(lk=4, ln=2, method="mle")
    assert I3Dc.intrinsic_dim == pytest.approx(4.575392144773673)

    ks, pv = I3Dc.model_validation_full(cdf=False)
    assert pv == pytest.approx(0.999999999)

    a, b = I3Dc.return_id_scaling(range(2, 10), method="mle", plot=False)
    d = np.array(
        [2.414214, 2.541382, 4.575392, 4.475622, 5.558403, 5.425578, 4.857541, 4.843088]
    )
    assert a == pytest.approx(d)

    a, b, c = I3Dc.return_id_scaling_k(range(5, 35, 5), plot=False)
    d = np.array([5.148287, 4.976629, 5.049483, 5.01851, 5.018707, 4.990721])
    assert a == pytest.approx(d)
