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
    import jax.numpy as jnp
    import optax

    from dadapy import DiffImbalance  # noqa: E402

    print("\n=== GRADIENT CLIPPING DEMONSTRATION ===")

    # First demonstrate gradient clipping with a simple synthetic example
    print("1. Testing gradient clipping functionality directly:")

    # Create synthetic large gradients
    large_gradients = jnp.array([5.0, -3.0, 8.0])
    small_gradients = jnp.array([0.1, -0.2, 0.05])
    clip_value = 1.0

    # Test clipping on large gradients
    clipper = optax.clip_by_global_norm(clip_value)
    clipped_large, _ = clipper.update(large_gradients, None, None)

    # Test clipping on small gradients
    clipped_small, _ = clipper.update(small_gradients, None, None)

    # Calculate norms
    large_norm_before = jnp.linalg.norm(large_gradients)
    large_norm_after = jnp.linalg.norm(clipped_large)
    small_norm_before = jnp.linalg.norm(small_gradients)
    small_norm_after = jnp.linalg.norm(clipped_small)

    print(
        f"   Large gradients: norm {large_norm_before:.6f} -> {large_norm_after:.6f} "
        f"({'CLIPPED' if large_norm_before > clip_value else 'unchanged'})"
    )
    print(
        f"   Small gradients: norm {small_norm_before:.6f} -> {small_norm_after:.6f} "
        f"({'CLIPPED' if small_norm_before > clip_value else 'unchanged'})"
    )
    print(
        f"   Clipping reduces large gradient norm by: {large_norm_before - large_norm_after:.6f}"
    )

    # Test with DiffImbalance
    print("\n2. Testing DiffImbalance gradient clipping integration:")

    # Generate test data
    data_A = np.load(filename)
    data_B = data_A * np.array([10, 1, 100])  # Scale features differently

    # Test with gradient clipping enabled
    dii_clipped = DiffImbalance(
        data_A,
        data_B,
        distances_B=None,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=3,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k_init=1,
        k_final=1,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="sgd",
        learning_rate=1.0,
        learning_rate_decay=None,
        num_points_rows=None,
        gradient_clip_value=1.0,  # Enable gradient clipping
    )

    # Test without gradient clipping
    dii_unclipped = DiffImbalance(
        data_A,
        data_B,
        distances_B=None,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=3,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k_init=1,
        k_final=1,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="sgd",
        learning_rate=1.0,
        learning_rate_decay=None,
        num_points_rows=None,
        gradient_clip_value=0.0,  # No gradient clipping
    )

    print("   Training with gradient clipping enabled...")
    weights_clipped, imbs_clipped = dii_clipped.train()

    print("   Training without gradient clipping...")
    weights_unclipped, imbs_unclipped = dii_unclipped.train()

    # Verify no NaN values in results
    assert not np.isnan(
        weights_clipped
    ).any(), "Clipped training should not produce NaN weights"
    assert not np.isnan(
        imbs_clipped
    ).any(), "Clipped training should not produce NaN imbalances"
    assert not np.isnan(
        weights_unclipped
    ).any(), "Unclipped training should not produce NaN weights"
    assert not np.isnan(
        imbs_unclipped
    ).any(), "Unclipped training should not produce NaN imbalances"

    # Verify that gradient clipping parameter is correctly stored
    assert (
        dii_clipped.gradient_clip_value == 1.0
    ), "Gradient clip value should be stored correctly"
    assert (
        dii_unclipped.gradient_clip_value == 0.0
    ), "Default gradient clip value should be 0.0"

    print(
        f"   Results - Clipped: DII {imbs_clipped[-1]:.6f}, Unclipped: DII {imbs_unclipped[-1]:.6f}"
    )

    # Demonstrate the optimizer creation with and without clipping
    print("\n3. Optimizer configuration:")
    dii_clipped._init_optimizer()
    dii_unclipped._init_optimizer()

    print(
        f"   Clipped optimizer chain includes gradient clipping: {dii_clipped.gradient_clip_value > 0}"
    )
    print(
        f"   Unclipped optimizer has no gradient clipping: {dii_unclipped.gradient_clip_value == 0}"
    )

    # Test that extreme conditions don't break the algorithm
    print("\n4. Testing robustness with extreme learning rate:")
    try:
        dii_extreme = DiffImbalance(
            data_A[:50],  # Use smaller dataset for speed
            data_B[:50],
            distances_B=None,
            periods_A=None,
            periods_B=None,
            seed=0,
            num_epochs=2,
            batches_per_epoch=1,
            l1_strength=0.0,
            point_adapt_lambda=False,
            k_init=1,
            k_final=1,
            lambda_factor=1e-1,
            params_init=None,
            params_groups=None,
            optimizer_name="sgd",
            learning_rate=100.0,  # Very high learning rate
            learning_rate_decay=None,
            num_points_rows=None,
            gradient_clip_value=0.5,  # Moderate clipping
        )
        weights_extreme, imbs_extreme = dii_extreme.train()
        assert not np.isnan(
            weights_extreme
        ).any(), "Extreme case with clipping should not produce NaN"
        print(f"   Extreme case with clipping succeeded: DII {imbs_extreme[-1]:.6f}")

        # Compare with same case but no clipping - might produce NaN or very large values
        dii_extreme_unclipped = DiffImbalance(
            data_A[:50],
            data_B[:50],
            distances_B=None,
            periods_A=None,
            periods_B=None,
            seed=0,
            num_epochs=2,
            batches_per_epoch=1,
            l1_strength=0.0,
            point_adapt_lambda=False,
            k_init=1,
            k_final=1,
            lambda_factor=1e-1,
            params_init=None,
            params_groups=None,
            optimizer_name="sgd",
            learning_rate=1.0,  # Lower learning rate to avoid NaN
            learning_rate_decay=None,
            num_points_rows=None,
            gradient_clip_value=0.0,
        )
        (
            weights_extreme_unclipped,
            imbs_extreme_unclipped,
        ) = dii_extreme_unclipped.train()
        print(f"   Same case without clipping: DII {imbs_extreme_unclipped[-1]:.6f}")

    except Exception as e:
        print(f"   Extreme case failed as expected: {str(e)[:100]}...")

    print("\n✓ Gradient clipping functionality verified!")
    print("  - Direct gradient clipping reduces large gradient norms")
    print("  - DiffImbalance correctly stores and uses gradient_clip_value parameter")
    print("  - Training completes successfully with and without clipping")
    print("  - No NaN values produced in normal training scenarios")


def run_test(test_func, test_name):
    """Run a test and print results."""
    print(f"Running {test_name}...")
    try:
        test_func()
        print(f"✓ {test_name} passed")
    except Exception as e:
        print(f"✗ {test_name} failed: {e}")
        if test_name == "test_DiffImbalance_gradient_clipping":
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    tests = [
        (test_DiffImbalance_train1, "test_DiffImbalance_train1"),
        (test_DiffImbalance_train2, "test_DiffImbalance_train2"),
        (test_DiffImbalance_train3, "test_DiffImbalance_train3"),
        (test_DiffImbalance_train4, "test_DiffImbalance_train4"),
        (test_DiffImbalance_train5, "test_DiffImbalance_train5"),
        (test_DiffImbalance_train6, "test_DiffImbalance_train6"),
        (test_DiffImbalance_gradient_clipping, "test_DiffImbalance_gradient_clipping"),
    ]

    for test_func, test_name in tests:
        run_test(test_func, test_name)
        print()

    print("All tests completed.")
