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


@pytest.mark.parametrize(
    "density_method, interpolator_method",
    [
        pytest.param(
            "compute_density_kstarNN",
            "return_interpolated_density_kstarNN",
            id="kstarNN",
        ),
        pytest.param(
            "compute_density_PAk",
            "return_interpolated_density_PAk",
            id="PAk",
        ),
    ],
)
@pytest.mark.parametrize("maxk", [4, 10])
def test_adaptive_density_interpolators_with_explicit_maxk(
    density_method, interpolator_method, maxk
):
    """Test adaptive interpolators with an explicit neighbour cap."""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")
    X = np.load(filename)[:25]
    kwargs = {"alpha": 0.05}

    reference = DensityEstimation(coordinates=X, maxk=maxk)
    expected_log_den, expected_log_den_err = getattr(reference, density_method)(
        **kwargs
    )

    de = DensityEstimation(coordinates=X)
    log_den, log_den_err, kstar = getattr(de, interpolator_method)(
        X, maxk=maxk, return_kstar=True, **kwargs
    )

    assert log_den == pytest.approx(expected_log_den)
    assert log_den_err == pytest.approx(expected_log_den_err)
    assert np.array_equal(kstar, reference.kstar)
    assert np.all(kstar >= 3)


@pytest.mark.parametrize(
    "interpolator_method",
    ["return_interpolated_density_kstarNN", "return_interpolated_density_PAk"],
)
def test_adaptive_density_interpolators_reject_unavailable_maxk(
    interpolator_method,
):
    """Test that the requested neighbour cap is available in the reference data."""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")
    X = np.load(filename)[:25]
    de = DensityEstimation(coordinates=X, maxk=10)

    with pytest.raises(ValueError, match="greater than the available maxk"):
        getattr(de, interpolator_method)(X, maxk=11)


@pytest.mark.parametrize(
    "interpolator_method",
    ["return_interpolated_density_kstarNN", "return_interpolated_density_PAk"],
)
@pytest.mark.parametrize("maxk", [1, 2, 3])
def test_adaptive_density_interpolators_reject_maxk_below_minimum(
    interpolator_method, maxk
):
    """Test the minimum neighbour cap used for automatic kstar selection."""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")
    X = np.load(filename)[:25]
    de = DensityEstimation(coordinates=X)

    with pytest.raises(ValueError, match="maxk must be at least 4"):
        getattr(de, interpolator_method)(X, maxk=maxk)
