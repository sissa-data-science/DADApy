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

"""Module for testing the density interpolators."""

import os

import numpy as np
import pytest

from dadapy import DensityEstimation


@pytest.mark.parametrize(
    "density_method, interpolator_method, kwargs",
    [
        pytest.param(
            "compute_density_kNN",
            "return_interpolated_density_kNN",
            {"k": 5},
            id="kNN",
        ),
        pytest.param(
            "compute_density_kstarNN",
            "return_interpolated_density_kstarNN",
            {"alpha": 0.05},
            id="kstarNN",
        ),
        pytest.param(
            "compute_density_PAk",
            "return_interpolated_density_PAk",
            {"alpha": 0.05},
            id="PAk",
        ),
        pytest.param(
            "compute_density_kstarNN",
            "return_interpolated_density_kstarNN",
            {
                "alpha": 0.05,
                "bonferroni_deloc": True,
                "bonferroni_loc": True,
            },
            id="kstarNN-bonferroni",
        ),
        pytest.param(
            "compute_density_PAk",
            "return_interpolated_density_PAk",
            {
                "alpha": 0.05,
                "bonferroni_deloc": True,
                "bonferroni_loc": True,
            },
            id="PAk-bonferroni",
        ),
    ],
)
def test_density_interpolators(density_method, interpolator_method, kwargs):
    """Test interpolated estimates evaluated on the reference data."""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")
    X = np.load(filename)[:25]

    de = DensityEstimation(coordinates=X)
    expected_log_den, expected_log_den_err = getattr(de, density_method)(**kwargs)
    expected_kstar = de.kstar.copy()

    interpolator = getattr(de, interpolator_method)
    result = interpolator(X, **kwargs)
    result_with_kstar = interpolator(X, return_kstar=True, **kwargs)

    assert len(result) == 2
    assert len(result_with_kstar) == 3
    assert result[0] == pytest.approx(expected_log_den)
    assert result[1] == pytest.approx(expected_log_den_err)
    assert result_with_kstar[0] == pytest.approx(result[0])
    assert result_with_kstar[1] == pytest.approx(result[1])
    assert np.array_equal(result_with_kstar[2], expected_kstar)
