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

"""Module for testing selection algorithms utilising differentiable information imbalance."""

import numpy as np
import pytest

from dadapy import Data, FeatureSelection

rng = np.random.default_rng()

def test_optimise_imbalance_typing():
    data = rng.random((10, 5))

    for period in [
        [3],
        "faz",
        np.array([2, 2], dtype=np.int8),
        np.array([2, 2], dtype=np.float32),
    ]:
        feature_selection = FeatureSelection(data, period=period)
        with pytest.raises(ValueError):
            feature_selection.optimize_kernel_imbalance(Data(data), 1)

    for initial_gammas in [np.array([2, 2], np.float32), ["faz"]]:
        feature_selection = FeatureSelection(data)
        with pytest.raises(ValueError):
            feature_selection.optimize_kernel_imbalance(
                Data(data), initial_gammas=initial_gammas
            )

def test_dist_matrix():
    data = rng.random((10, 5))
    feature_selection = FeatureSelection(data)
    assert feature_selection._full_distance_matrix is None
