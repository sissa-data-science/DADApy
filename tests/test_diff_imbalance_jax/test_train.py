# Copyright 2021-2024 The DADApy Authors. All Rights Reserved.
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

"""Module for testing methods of the DiffImbalance class."""

import os
import sys

import numpy as np
import pytest
from jax import config

config.update("jax_platform_name", "cpu")
filename = os.path.join(os.path.split(__file__)[0], "../3d_gauss_small_z_var.npy")


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train1():
    """Test DII train function."""
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A
    print(f"Ground truth weights = {weights_ground_truth}\n")

    expected_weights = [0.15569, 0.0724, 0.02274]
    expected_imb = 0.04744

    # train the DII to recover ground-truth metric
    dii = DiffImbalance(
        data_A,  # matrix of shape (N,D_A)
        data_B,  # matrix of shape (N,D_B)
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k_init=None,
        k_final=None,
        lambda_init=1e-3,
        lambda_final=1e-3,
        lambda_factor=1e-1,
        init_params=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        compute_error=False,
        ratio_rows_columns=1,
        num_points_rows=None,
        discard_close_ind=5,
    )
    weights, imbs = dii.train()

    assert weights[-1] == pytest.approx(expected_weights, abs=0.001)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.001)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train2():
    """Test DII train function."""
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    init_params = np.array([10.0, 10.0, 10.0])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A
    print(f"Ground truth weights = {weights_ground_truth}\n")

    expected_weights = [10.90023, 5.43393, 12.31492]
    expected_imb = 0.04298

    # train the DII
    dii = DiffImbalance(
        data_A,
        data_B,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=5,
        l1_strength=1e-4,
        point_adapt_lambda=False,
        k_init=1,
        k_final=1,
        lambda_init=None,
        lambda_final=None,
        lambda_factor=1e-1,
        init_params=init_params,
        optimizer_name="adam",
        learning_rate=1e-1,
        learning_rate_decay=None,
        compute_error=False,
        ratio_rows_columns=1,
        num_points_rows=None,
        discard_close_ind=None,
    )
    weights, imbs = dii.train()

    assert weights[-1] == pytest.approx(expected_weights, abs=0.001)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.001)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train3():
    """Test DII train function."""
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A
    print(f"Ground truth weights = {weights_ground_truth}\n")

    expected_weights = [0.12776, 0.09972, 0.06112]
    expected_imb = 0.53706

    # train the DII
    dii = DiffImbalance(
        data_A,
        data_B,
        periods_A=2 * np.pi,
        periods_B=2 * np.pi,
        seed=0,
        num_epochs=10,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=True,
        k_init=1,
        k_final=1,
        lambda_init=None,
        lambda_final=None,
        lambda_factor=1e-1,
        init_params=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="exp",
        compute_error=True,
        ratio_rows_columns=1,
        num_points_rows=None,
        discard_close_ind=None,
    )
    weights, imbs = dii.train()

    assert weights[-1] == pytest.approx(expected_weights, abs=0.01)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.01)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train4():
    """Test DII train function."""
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10,3,100])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis,:] * data_A
    print(f"Ground truth weights = {weights_ground_truth}\n")

    expected_weights = [0.1312, 0.05073, 0.10106]
    expected_imb = 0.040379

    # train the DII to recover ground-truth metric
    dii = DiffImbalance(
        data_A, # matrix of shape (N,D_A)
        data_B, # matrix of shape (N,D_B)
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k_init=1,
        k_final=1,
        lambda_init=None,
        lambda_final=None,
        lambda_factor=1e-1,
        init_params=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        compute_error=False,
        ratio_rows_columns=1,
        num_points_rows=50,
        discard_close_ind=None
    )
    weights, imbs = dii.train()

    # scale learnt weights in same range of ground-truth ones (same magnitude of the largest one)
    print(f"Learnt weights: {weights[-1]}")
    print(f"Final imb: {imbs[-1]}")

    assert weights[-1] == pytest.approx(expected_weights, abs=0.01)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.01)

