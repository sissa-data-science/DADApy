# Copyright 2021-2025 The DADApy Authors. All Rights Reserved.
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

config.update("jax_platforms", "cpu")
filename = os.path.join(os.path.split(__file__)[0], "../3d_gauss_small_z_var.npy")


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train1():
    """Test DII train function."""
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    expected_weights = [0.77552, 0.268687, 0.571294]
    expected_imb = 0.055127
    expected_imb_final = 0.055127

    # train the DII to recover ground-truth metric
    dii = DiffImbalance(
        data_A,  # matrix of shape (N,D_A)
        data_B,  # matrix of shape (N,D_B)
        distances_B=None,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k=10,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="sgd",
        learning_rate=1.0,
        learning_rate_decay="cos",
    )
    weights, imbs = dii.train()

    # compute final DII
    imb_final = dii.return_final_dii()

    assert weights[-1] == pytest.approx(expected_weights, abs=0.001)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.001)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train2():
    """Test DII train function."""
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    params_init = np.array([10.0, 10.0, 10.0])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    expected_weights = [16.898588, -3.799701, 0.0]
    expected_imb = 0.131398
    expected_imb_final = 0.048345

    # train the DII
    dii = DiffImbalance(
        data_A,
        data_B,
        distances_B=None,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=5,
        discard_close_ind=3,
        l1_strength=1e-4,
        point_adapt_lambda=False,
        k=1,
        lambda_factor=1e-1,
        params_init=params_init,
        params_groups=None,
        optimizer_name="sgd",  # the L1 regularization is supported only with SGD
        learning_rate=300.0,  # the SGD step on the direction of the weights scales as lr / |params_init|^2
        learning_rate_decay=None,
    )
    weights, imbs = dii.train()

    imb_final = dii.return_final_dii()

    assert weights[-1] == pytest.approx(expected_weights, abs=0.001)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.001)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train3():
    """Test DII train function."""
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    expected_weights = [0.772524, 0.369993, 0.516055]
    expected_imb = 0.431966
    expected_imb_final = 0.431966

    # train the DII
    dii = DiffImbalance(
        data_A,
        data_B,
        distances_B=None,
        periods_A=2 * np.pi,
        periods_B=2 * np.pi,
        seed=0,
        num_epochs=10,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=True,
        k=1,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="sgd",
        learning_rate=1.0,
        learning_rate_decay="cos",
    )
    weights, imbs = dii.train()

    # compute final DII
    imb_final = dii.return_final_dii()

    assert weights[-1] == pytest.approx(expected_weights, abs=0.01)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.01)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train4():
    """Test DII train function."""
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    expected_weights = [0.715982, 0.391455, 0.578043]
    expected_imb = 0.035573
    expected_imb_final = 0.035573

    # train the DII to recover ground-truth metric
    dii = DiffImbalance(
        data_A,  # matrix of shape (N,D_A)
        data_B,  # matrix of shape (N,D_B)
        distances_B=None,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=1,
        discard_close_ind=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k=1,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="sgd",
        learning_rate=1.0,
        learning_rate_decay="cos",
    )
    weights, imbs = dii.train()

    # compute final DII
    imb_final = dii.return_final_dii()

    assert weights[-1] == pytest.approx(expected_weights, abs=0.01)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.01)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train5():
    """Test DII train function."""
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A
    params_init = [1, 0.1]
    params_groups = [2, 1]

    expected_weights = [0.999981, 0.100193]
    expected_imb = 0.046835
    expected_imb_final = 0.046835

    # train the DII to recover ground-truth metric
    dii = DiffImbalance(
        data_A,  # matrix of shape (N,D_A)
        data_B,  # matrix of shape (N,D_B)
        distances_B=None,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=1,
        discard_close_ind=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k=1,
        lambda_factor=1e-1,
        params_init=params_init,
        params_groups=params_groups,
        optimizer_name="sgd",
        learning_rate=1.0,
        learning_rate_decay="cos",
    )
    weights, imbs = dii.train()

    # compute final DII
    imb_final = dii.return_final_dii()

    assert weights[-1] == pytest.approx(expected_weights, abs=0.01)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.01)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train6():
    """Test DII train function."""
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A
    distances_B = ((data_B[np.newaxis, :, :] - data_B[:, np.newaxis, :]) ** 2).sum(
        axis=-1
    )

    expected_weights = [0.715982, 0.391455, 0.578043]
    expected_imb = 0.035573
    expected_imb_final = 0.035573

    # train the DII to recover ground-truth metric
    dii = DiffImbalance(
        data_A,  # matrix of shape (N,D_A)
        data_B=None,  # matrix of shape (N,D_B)
        distances_B=distances_B,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=1,
        discard_close_ind=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k=1,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="sgd",
        learning_rate=1.0,
        learning_rate_decay="cos",
    )
    weights, imbs = dii.train()

    # compute final DII
    imb_final = dii.return_final_dii()

    assert weights[-1] == pytest.approx(expected_weights, abs=0.01)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.01)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)
