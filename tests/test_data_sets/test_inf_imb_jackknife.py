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

"""Module for testing information imbalance jackknife computation."""

import os

import numpy as np

from dadapy import DataSets
from dadapy.metric_comparisons import MetricComparisons

filename = os.path.join(os.path.split(__file__)[0], "../3d_gauss_small_z_var.npy")


def test_information_imbalance_jackknife():
    X = np.load(filename)[:100, :]

    d1 = MetricComparisons(coordinates=X[:, [0]], maxk=X.shape[0] - 1)
    d2 = MetricComparisons(
        coordinates=X[
            :,
            [
                1,
            ],
        ],
        maxk=X.shape[0] - 1,
    )

    ds = DataSets()

    mean, std = ds.return_inf_imb_jackknife_d1_to_d2(d1, d2, file_name=None)

    assert np.isclose(mean, 1, atol=3 * std)
    assert np.isclose(mean, 1.0176471788593)
