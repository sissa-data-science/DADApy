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

"""Module for testing the Gride ID estimator with kstar estimation of the optimal scale."""

import numpy as np
import pytest

from dadapy import Data


def test_compute_id_kstar_gride():
    """Test that the id estimations with Gride and kstar work correctly."""

    np.random.seed(0)

    X = np.random.normal(0, 1, size=(1000, 2))

    de = Data(coordinates=X, maxk=X.shape[0] - 1)

    ids, ids_err, kstars, log_likelihoods = de.return_ids_kstar_gride(
        initial_id=5, n_iter=3
    )

    assert ids == pytest.approx([2.02, 1.93, 1.93], abs=0.01)
    assert log_likelihoods == pytest.approx([668.1, 1153.89, 1170.77], abs=0.01)
