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

"""Module for testing differentiable information imbalance methods."""

import itertools as it

import numpy as np

import dadapy._cython.cython_differentiable_imbalance as c_dii
from dadapy._utils import differentiable_imbalance as dii

rng = np.random.default_rng()


def test_typing():
    """Test typing decorator function"""
    data = rng.random((10, 5))
    for dtype in [np.float16, np.float32, np.double, np.float64, np.float128]:
        # This should run with only floats as input arrays
        dist_mat = dii._return_full_dist_matrix(
            data.astype(dtype), period=np.zeros(5, dtype=dtype), njobs=1, cythond=True
        )
        assert dist_mat.dtype == dii.CYTHON_DTYPE

    @dii.cast_ndarrays
    def cast_func(*args, **kwargs):
        return args, kwargs

    assert cast_func(0)[0][0] == 0
    assert cast_func("string")[0][0] == "string"
    assert cast_func(np.zeros(10, dtype=np.int32))[0][0].dtype == np.int32
    assert cast_func(np.zeros(10, dtype=np.int64))[0][0].dtype == np.int64

    assert cast_func(np.zeros(10, dtype=np.float16))[0][0].dtype == dii.CYTHON_DTYPE
    assert cast_func(np.zeros(10, dtype=np.float128))[0][0].dtype == dii.CYTHON_DTYPE
    assert cast_func(np.zeros(10, dtype=np.double))[0][0].dtype == dii.CYTHON_DTYPE

    assert cast_func(0, test=np.zeros(10, dtype=np.float16))[0][0] == 0
    assert (
        cast_func(0, test=np.zeros(10, dtype=np.float16))[1]["test"].dtype
        == dii.CYTHON_DTYPE
    )


def test_dist_matrix():
    """Test proper return shape of full distance matrix for all cases"""
    for n_data in np.logspace(1, 2, 10, dtype=np.int16):
        for n_dim in np.logspace(1, 2, 10, dtype=np.int16):
            data = rng.random((n_data, n_dim))
            for period, cythond in it.product(
                [None, np.ones(n_dim)], [False, True]
            ):  # TODO: should int, float also be accepted for period?
                dist_mat = dii._return_full_dist_matrix(
                    data=data, period=period, cythond=cythond, njobs=1
                )
                assert dist_mat.shape[0] == n_data
                assert dist_mat.shape[1] == n_data


def test_rank_matrix():
    """Test full rank matrix shaping and make a simple test for quadratic data.
    Nearest neighbor should always be previous data point here.
    """
    for n_data in np.logspace(1, 2, 10, dtype=np.int16):
        for n_dim in np.logspace(1, 2, 10, dtype=np.int16):
            # Construct data so that nearest point is always previous one
            data = np.zeros((n_data, n_dim), dtype=np.float64)
            data[:, :] = np.arange(n_data)[:, np.newaxis] ** 2

            for cythond in [False, True]:
                ranks = dii._return_full_rank_matrix(
                    data=data,
                    period=np.zeros((data.shape[-1]), dtype=np.double),
                    cythond=cythond,
                    njobs=1,
                )
                assert ranks.shape[0] == n_data
                assert ranks.shape[1] == n_data

                assert (
                    np.count_nonzero(np.diag(ranks, k=-1) == 1) == n_data - 1
                ), "Nearest neighbours not properly calculated."


def test_py_kernel_gradient():
    for n_data in np.logspace(1, 2, 10, dtype=np.int16):
        for n_dim in np.logspace(1, 2, 10, dtype=np.int16):
            njobs = 4
            gammas = rng.random((n_dim,), dtype=dii.CYTHON_DTYPE)
            data = rng.random((n_data, n_dim), dtype=dii.CYTHON_DTYPE)
            dist_mat = dii._return_full_dist_matrix(
                data * gammas,
                period=np.zeros(n_dim, dtype=dii.CYTHON_DTYPE),
                njobs=njobs,
                cythond=False,
            )
            ranks = dii._return_full_rank_matrix(
                data,
                period=np.zeros(n_dim, dtype=dii.CYTHON_DTYPE),
                njobs=njobs,
                cythond=False,
            )

            py_grad = dii._return_dii_gradient_python(
                dists_rescaled_A=dist_mat,
                data_A=data * gammas,
                rank_matrix_B=ranks,
                gammas=gammas,
                lambd=1.0,
                period=np.zeros(n_dim, dtype=dii.CYTHON_DTYPE),
                njobs=njobs,
            )

            cython_grad = c_dii.return_dii_gradient_cython(
                dist_mat,
                data * gammas,
                ranks,
                gammas,
                1.0,
                np.zeros(n_dim, dtype=dii.CYTHON_DTYPE),
                njobs,
                False,
            )

            # TODO: Is it okay for these to be so "far apart".
            assert cython_grad.shape[0] == n_dim
            assert np.allclose(cython_grad, py_grad, atol=1e-1, rtol=1e-2)
            # assert np.allclose(cython_grad, fast_pygrad, atol=1e-1, rtol=1e-2)
