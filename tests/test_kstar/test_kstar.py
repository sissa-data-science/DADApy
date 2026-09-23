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

"""Module for testing the KStar class."""

import numpy as np
import pytest

from dadapy import KStar

try:
    import jax  # noqa: F401

    HAS_JAX = True
except ModuleNotFoundError:
    HAS_JAX = False

# define a basic dataset with 6 points
data = np.array([[0.01, 0], [0.12, 0], [0.23, 0], [4.0, 0], [4.11, 0], [4.22, 0]])
rng = np.random.default_rng(42)
backend_data = np.concatenate(
    (rng.normal(-1.0, 0.7, size=(40, 2)), rng.normal(1.0, 0.7, size=(40, 2)))
)


expected_kstar_high_alpha = np.array([3, 3, 3, 3, 3, 3])
expected_kstar_low_alpha = np.array([4, 4, 4, 4, 4, 4])


def test_compute_kstar_high_alpha():
    """Test kstar with a permissive significance level."""
    # create the KStar object
    kstar = KStar(coordinates=data, n_jobs=1)
    # compute kstar
    kstar.compute_kstar(alpha=0.999999999)
    # check that the result is correct
    assert np.array_equal(kstar.kstar, expected_kstar_high_alpha)


def test_compute_kstar_low_alpha():
    """Test kstar with a stringent significance level."""
    # create the KStar object
    kstar = KStar(coordinates=data, n_jobs=1)
    # compute kstar
    kstar.compute_kstar(alpha=1e-300)
    # check that the result is correct
    assert np.array_equal(kstar.kstar, expected_kstar_low_alpha)


def test_set_kstar():
    """Test the set_kstar method."""
    # create the KStar object
    kstar = KStar(coordinates=data, n_jobs=1)
    # set kstar
    set_kstar = [1, 2, 3, 4, 5, 6]
    kstar.set_kstar(k=set_kstar)
    # check that the result is correct
    assert np.array_equal(kstar.kstar, set_kstar)


def test_compute_kstar_auto_backend():
    """Test that auto backend is consistent with cython backend."""
    kstar = KStar(coordinates=backend_data, maxk=30, n_jobs=1)
    kstar.compute_kstar(alpha=0.05, backend="cython")
    expected = kstar.kstar.copy()
    kstar.compute_kstar(alpha=0.05, backend="auto")
    assert np.array_equal(kstar.kstar, expected)


@pytest.mark.parametrize("maxk", [1, 2, 3, 4])
def test_compute_kstar_backends_small_maxk(maxk):
    """Test backend agreement when no likelihood-ratio test can be performed."""
    cython = KStar(coordinates=backend_data, maxk=maxk, n_jobs=1)
    cython.set_id(2.0)
    cython.compute_kstar(alpha=0.05, backend="cython")

    parallel = KStar(coordinates=backend_data, maxk=maxk, n_jobs=2)
    parallel.set_id(2.0)
    parallel.compute_kstar(alpha=0.05, backend="cython")

    assert np.array_equal(parallel.kstar, cython.kstar)

    if HAS_JAX:
        jax_kstar = KStar(coordinates=backend_data, maxk=maxk, n_jobs=1)
        jax_kstar.set_id(2.0)
        jax_kstar.compute_kstar(alpha=0.05, backend="jax")
        assert np.array_equal(jax_kstar.kstar, cython.kstar)


@pytest.mark.parametrize(
    "bonferroni_deloc, bonferroni_loc",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_compute_kstar_parallel_backend(bonferroni_deloc, bonferroni_loc):
    """Test that serial and parallel Cython implementations agree."""
    serial = KStar(coordinates=backend_data, maxk=30, n_jobs=1)
    serial.compute_kstar(
        alpha=0.05,
        bonferroni_deloc=bonferroni_deloc,
        bonferroni_loc=bonferroni_loc,
        backend="cython",
    )

    parallel = KStar(coordinates=backend_data, maxk=30, n_jobs=2)
    parallel.compute_kstar(
        alpha=0.05,
        bonferroni_deloc=bonferroni_deloc,
        bonferroni_loc=bonferroni_loc,
        backend="cython",
    )

    assert np.array_equal(parallel.kstar, serial.kstar)


@pytest.mark.parametrize(
    "bonferroni_deloc, bonferroni_loc",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_compute_kstar_jax_backend(bonferroni_deloc, bonferroni_loc):
    """Test the JAX backend or the expected error if JAX is unavailable."""
    kstar = KStar(coordinates=backend_data, maxk=30, n_jobs=1)
    kwargs = {
        "alpha": 0.05,
        "bonferroni_deloc": bonferroni_deloc,
        "bonferroni_loc": bonferroni_loc,
    }
    if HAS_JAX:
        kstar.compute_kstar(backend="cython", **kwargs)
        expected = kstar.kstar.copy()
        kstar.compute_kstar(backend="jax", **kwargs)
        assert np.array_equal(kstar.kstar, expected)
    else:
        with pytest.raises(ModuleNotFoundError):
            kstar.compute_kstar(backend="jax", **kwargs)


def test_compute_kstar_bonferroni_deloc():
    """Test the correction for simultaneous tests across data points."""
    kstar_uncorrected = KStar(coordinates=backend_data, maxk=30, n_jobs=1)
    kstar_uncorrected.compute_kstar(alpha=0.05, bonferroni_deloc=False)

    kstar_corrected = KStar(coordinates=backend_data, maxk=30, n_jobs=1)
    kstar_corrected.compute_kstar(alpha=0.05, bonferroni_deloc=True)

    assert np.all(kstar_corrected.kstar >= kstar_uncorrected.kstar)
    assert np.any(kstar_corrected.kstar > kstar_uncorrected.kstar)


def test_compute_kstar_bonferroni_loc():
    """Test the correction for successive neighbourhood tests."""
    kstar_uncorrected = KStar(coordinates=backend_data, maxk=30, n_jobs=1)
    kstar_uncorrected.compute_kstar(alpha=0.05, bonferroni_loc=False)

    kstar_corrected = KStar(coordinates=backend_data, maxk=30, n_jobs=1)
    kstar_corrected.compute_kstar(alpha=0.05, bonferroni_loc=True)

    assert np.all(kstar_corrected.kstar >= kstar_uncorrected.kstar)
    assert np.any(kstar_corrected.kstar > kstar_uncorrected.kstar)


def test_compute_kstar_bonferroni_both():
    """Test with both bonferroni corrections."""
    kstar = KStar(coordinates=backend_data, maxk=30, n_jobs=1)
    kstar.compute_kstar(alpha=0.05, bonferroni_deloc=True, bonferroni_loc=True)

    assert kstar.kstar is not None
    assert len(kstar.kstar) == len(backend_data)
    assert np.all(kstar.kstar > 0)
