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

    expected_weights = [0.13817, 0.04679, 0.09338]
    expected_imb = 0.03670
    expected_imb_final = 0.03670

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
        k_init=10,
        k_final=1,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        num_points_rows=None,
        gradient_clip_value=0.0,
    )
    weights, imbs = dii.train()

    # compute final DII
    imb_final, _ = dii.return_final_dii(
        compute_error=False, ratio_rows_columns=None, seed=0, discard_close_ind=0
    )

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

    expected_weights = [12.03902, 5.10201, 11.3592]
    expected_imb = 0.11381
    expected_imb_final = 0.05178

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
        l1_strength=1e-4,
        point_adapt_lambda=False,
        k_init=1,
        k_final=1,
        lambda_factor=1e-1,
        params_init=params_init,
        params_groups=None,
        optimizer_name="adam",
        learning_rate=1e-1,
        learning_rate_decay=None,
        gradient_clip_value=0.0,
    )
    weights, imbs = dii.train()

    imb_final, _ = dii.return_final_dii(
        compute_error=False, ratio_rows_columns=None, seed=0, discard_close_ind=10
    )

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

    expected_weights = [0.16791, 0.04188, 0.00716]
    expected_imb = 0.438795
    expected_imb_final = 0.59615
    expected_error_final = 0.07216

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
        k_init=1,
        k_final=1,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="exp",
        num_points_rows=None,
        gradient_clip_value=0.0,
    )
    weights, imbs = dii.train()

    # compute final DII
    imb_final, error_final = dii.return_final_dii(
        compute_error=True, ratio_rows_columns=1, seed=0, discard_close_ind=0
    )

    assert weights[-1] == pytest.approx(expected_weights, abs=0.01)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.01)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)
    assert error_final == pytest.approx(expected_error_final, abs=0.001)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train4():
    """Test DII train function."""
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    expected_weights = [0.1312, 0.05073, 0.10106]
    expected_imb = 0.0403795
    expected_imb_final = 0.08504
    expected_error_final = 0.01226

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
        k_init=1,
        k_final=1,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        num_points_rows=50,
        gradient_clip_value=0.0,
    )
    weights, imbs = dii.train()

    # compute final DII
    imb_final, error_final = dii.return_final_dii(
        compute_error=True, ratio_rows_columns=0.5, seed=0, discard_close_ind=1
    )

    assert weights[-1] == pytest.approx(expected_weights, abs=0.01)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.01)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)
    assert error_final == pytest.approx(expected_error_final, abs=0.001)


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

    expected_weights = [1, 0.10002]
    expected_imb = 0.05411
    expected_imb_final = 0.10039
    expected_error_final = 0.01771

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
        k_init=1,
        k_final=1,
        lambda_factor=1e-1,
        params_init=params_init,
        params_groups=params_groups,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        num_points_rows=50,
        gradient_clip_value=0.0,
    )
    weights, imbs = dii.train()

    # compute final DII
    imb_final, error_final = dii.return_final_dii(
        compute_error=True, ratio_rows_columns=0.5, seed=0, discard_close_ind=1
    )

    assert weights[-1] == pytest.approx(expected_weights, abs=0.01)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.01)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)
    assert error_final == pytest.approx(expected_error_final, abs=0.001)


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

    expected_weights = [0.1312, 0.05073, 0.10106]
    expected_imb = 0.0403795
    expected_imb_final = 0.08504
    expected_error_final = 0.01226

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
        l1_strength=0.0,
        point_adapt_lambda=False,
        k_init=1,
        k_final=1,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        num_points_rows=50,
        gradient_clip_value=0.0,
    )
    weights, imbs = dii.train()

    # compute final DII
    imb_final, error_final = dii.return_final_dii(
        compute_error=True, ratio_rows_columns=0.5, seed=0, discard_close_ind=1
    )

    assert weights[-1] == pytest.approx(expected_weights, abs=0.01)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.01)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)
    assert error_final == pytest.approx(expected_error_final, abs=0.001)

@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_gradient_clipping():
    """Test DII with gradient clipping to prevent NaN values."""
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data that might cause gradient explosion
    weights_ground_truth = np.array([100, 0.01, 1000])  # extreme weights
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    # Test with gradient clipping enabled
    dii_clipped = DiffImbalance(
        data_A,
        data_B,
        distances_B=None,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k_init=1,
        k_final=1,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="adam",
        learning_rate=1e-1,  # Higher learning rate to trigger potential instability
        learning_rate_decay=None,
        num_points_rows=None,
        gradient_clip_value=1.0,  # Enable gradient clipping
    )

    # Training should complete without NaN errors
    print("Starting clipped training...")
    weights_clipped, imbs_clipped = dii_clipped.train()
    print(f"Clipped training completed. Final weights shape: {weights_clipped[-1].shape}")
    print(f"Clipped training imbalances: {len(imbs_clipped)} values")

    # Test without gradient clipping for comparison
    print("Starting unclipped training...")
    dii_unclipped = DiffImbalance(
        data_A,
        data_B,
        distances_B=None,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k_init=1,
        k_final=1,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="adam",
        learning_rate=1e-2,  # Lower learning rate to prevent NaN
        learning_rate_decay=None,
        num_points_rows=None,
        gradient_clip_value=0.0,  # No gradient clipping
    )

    weights_unclipped, imbs_unclipped = dii_unclipped.train()
    print(f"Unclipped training completed. Final weights shape: {weights_unclipped[-1].shape}")
    print(f"Unclipped training imbalances: {len(imbs_unclipped)} values")

    # Verify no NaN values in results
    assert not np.isnan(weights_clipped).any(), "Clipped training should not produce NaN weights"
    assert not np.isnan(imbs_clipped).any(), "Clipped training should not produce NaN imbalances"
    assert not np.isnan(weights_unclipped).any(), "Unclipped training should not produce NaN weights"
    assert not np.isnan(imbs_unclipped).any(), "Unclipped training should not produce NaN imbalances"

    # Verify that gradient clipping parameter is correctly stored
    assert dii_clipped.gradient_clip_value == 1.0, "Gradient clip value should be stored correctly"
    assert dii_unclipped.gradient_clip_value == 0.0, "Default gradient clip value should be 0.0"

    # Verify training completed successfully
    print(f"Clipped training completed {len(weights_clipped)} weight arrays (expected 11 = num_epochs + 1)")
    print(f"Unclipped training completed {len(weights_unclipped)} weight arrays (expected 11 = num_epochs + 1)")
    
    # DiffImbalance.train() returns (num_epochs + 1) arrays (includes initial state)
    expected_length = 11  # 10 epochs + 1 initial state
    assert len(weights_clipped) == expected_length, f"Clipped training should return {expected_length} weight arrays, got {len(weights_clipped)}"
    assert len(weights_unclipped) == expected_length, f"Unclipped training should return {expected_length} weight arrays, got {len(weights_unclipped)}"

    # Test that the optimization is working (imbalance should generally decrease)
    print(f"Clipped training - Initial DII: {imbs_clipped[0]:.6f}, Final DII: {imbs_clipped[-1]:.6f}")
    print(f"Unclipped training - Initial DII: {imbs_unclipped[0]:.6f}, Final DII: {imbs_unclipped[-1]:.6f}")
    print(f"Clipped final weights: {weights_clipped[-1]}")
    print(f"Unclipped final weights: {weights_unclipped[-1]}")
    
    assert imbs_clipped[-1] < imbs_clipped[0], "DII should generally decrease during training"
    assert imbs_unclipped[-1] < imbs_unclipped[0], "DII should generally decrease during training"


if __name__ == "__main__":
    print("Running test_DiffImbalance_train1...")
    try:
        test_DiffImbalance_train1()
        print("✓ test_DiffImbalance_train1 passed")
    except Exception as e:
        print(f"✗ test_DiffImbalance_train1 failed: {e}")
    
    print("\nRunning test_DiffImbalance_train2...")
    try:
        test_DiffImbalance_train2()
        print("✓ test_DiffImbalance_train2 passed")
    except Exception as e:
        print(f"✗ test_DiffImbalance_train2 failed: {e}")
    
    print("\nRunning test_DiffImbalance_train3...")
    try:
        test_DiffImbalance_train3()
        print("✓ test_DiffImbalance_train3 passed")
    except Exception as e:
        print(f"✗ test_DiffImbalance_train3 failed: {e}")
    
    print("\nRunning test_DiffImbalance_train4...")
    try:
        test_DiffImbalance_train4()
        print("✓ test_DiffImbalance_train4 passed")
    except Exception as e:
        print(f"✗ test_DiffImbalance_train4 failed: {e}")
    
    print("\nRunning test_DiffImbalance_train5...")
    try:
        test_DiffImbalance_train5()
        print("✓ test_DiffImbalance_train5 passed")
    except Exception as e:
        print(f"✗ test_DiffImbalance_train5 failed: {e}")
    
    print("\nRunning test_DiffImbalance_train6...")
    try:
        test_DiffImbalance_train6()
        print("✓ test_DiffImbalance_train6 passed")
    except Exception as e:
        print(f"✗ test_DiffImbalance_train6 failed: {e}")
    
    print("\nRunning test_DiffImbalance_gradient_clipping...")
    try:
        test_DiffImbalance_gradient_clipping()
        print("✓ test_DiffImbalance_gradient_clipping passed")
    except Exception as e:
        print(f"✗ test_DiffImbalance_gradient_clipping failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nAll tests completed.")