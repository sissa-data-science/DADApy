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

import sys

import numpy as np
import pytest
from jax import config

config.update("jax_platform_name", "cpu")


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train1():
    """Test DII train function."""
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data
    np.random.seed(0)
    weights_ground_truth = np.array([10, 3, 1, 30, 7.3])
    data_A = np.random.normal(loc=0, scale=1.0, size=(200, 5))
    data_B = weights_ground_truth[np.newaxis, :] * data_A
    print(f"Ground truth weights = {weights_ground_truth}\n")

    expected_weights = [0.08612, 0.03023, 0.014, 0.19341, 0.06376]
    expected_imb = 0.01113

    # train the DII
    dii = DiffImbalance(
        data_A,
        data_B,
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
    weights_ground_truth = np.array([10, 3, 1, 30, 7.3])
    np.random.seed(0)
    init_params = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    data_A = np.random.normal(loc=0, scale=1.0, size=(200, 5))
    data_B = weights_ground_truth[np.newaxis, :] * data_A
    print(f"Ground truth weights = {weights_ground_truth}\n")

    expected_weights = [8.40182, 7.44459, 7.24351, 15.58236, 8.87181]
    expected_imb = 0.0427368

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
    weights_ground_truth = np.array([10, 3, 1, 30, 7.3])
    np.random.seed(0)
    data_A = np.random.normal(loc=0, scale=1.0, size=(200, 5))
    data_B = weights_ground_truth[np.newaxis, :] * data_A
    print(f"Ground truth weights = {weights_ground_truth}\n")

    expected_weights = [0.01044, 0.16464, 0.12139, 0.08465, 0.02972]
    expected_imb = 0.60103

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
        discard_close_ind=None,
    )
    weights, imbs = dii.train()

    assert weights[-1] == pytest.approx(expected_weights, abs=0.001)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.001)
