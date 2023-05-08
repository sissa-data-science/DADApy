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

"""Module for testing utils functions."""


import numpy as np
import pytest

from dadapy._utils import utils


def test_zero_dist():
    """Test that a warning message appear if there are overlapping datapoints."""
    X = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    with pytest.warns(UserWarning):
        utils.compute_nn_distances(X, maxk=2, metric="euclidean", period=None)
