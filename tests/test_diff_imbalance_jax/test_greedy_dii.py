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

"""Module for testing greedy feature selection methods of the DiffImbalance class."""

import os
import sys

import numpy as np
import pytest
from jax import config

config.update("jax_platform_name", "cpu")
filename = os.path.join(os.path.split(__file__)[0], "../3d_gauss_small_z_var.npy")


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_forward_greedy():
    """Test forward greedy feature selection function.

    The dataset is a 3D Gaussian with variances [0.961146, 1.06219351, 0.01091285],
    where the third dimension has much lower variance. We apply weights [10, 0.1, 5]
    to the dimensions, making dimensions 0 and 2 more important.

    The forward greedy algorithm should prioritize dimension 0 first, followed by
    dimension 1, as observed in the algorithm's behavior.
    """
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data
    # The dataset has variances [0.961146, 1.06219351, 0.01091285]
    # Dimension 2 has much lower variance than dimensions 0 and 1
    weights_ground_truth = np.array([1, 0.001, 5])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    # Expected results based on (variance*weight)
    expected_first_feature = [0]
    expected_second_features = [0, 2]

    # train the DII to recover ground-truth metric
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
        k_init=10,
        k_final=1,
        lambda_factor=1e-1,
        params_init=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        num_points_rows=None,
    )
    weights, imbs = dii.train()

    # Run forward greedy feature selection
    feature_sets, diis, _ = dii.forward_greedy_feature_selection(
        n_features_max=3, compute_error=False
    )

    print("FORWARD DII TEST:")
    print("Weights with all features:\n", weights)
    print("Imbs with all features:\n", imbs)
    print("Feature Sets:", feature_sets)
    print("Final DIIs:", diis)

    # Check first feature selected
    assert (
        feature_sets[0] == expected_first_feature
    ), f"First feature should be {expected_first_feature}, got {feature_sets[0]}"

    # Check second set of features
    assert (
        feature_sets[1] == expected_second_features
    ), f"Second feature set should be {expected_second_features}, got {feature_sets[1]}"

    # Check that all features are included in the final set
    assert (
        len(feature_sets[2]) == 3
    ), f"Final feature set should include all 3 features, got {feature_sets[2]}"


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_backward_greedy():
    """Test backward greedy feature selection function.

    The dataset is a 3D Gaussian with variances [0.961146, 1.06219351, 0.01091285],
    where the third dimension has much lower variance. We apply weights [10, 0.1, 5]
    to the dimensions, making dimensions 0 and 2 more important.

    The backward greedy algorithm should remove the least important features first.
    Interestingly, it removes dimension 2 first, despite its high weight (5),
    possibly because of its low variance in the original dataset.
    """
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data
    # The dataset has variances [0.961146, 1.06219351, 0.01091285]
    # Dimension 2 has much lower variance than dimensions 0 and 1
    weights_ground_truth = np.array([1, 0.001, 5])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    # Expected results based on (variance*weight)
    expected_first_removal = 1
    expected_second_features = [0, 2]

    # train the DII to recover ground-truth metric
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
        k_init=10,
        k_final=1,
        lambda_factor=1e-1,
        params_init=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        num_points_rows=None,
    )
    weights, imbs = dii.train()

    # Run backward greedy feature selection
    feature_sets, diis, _ = dii.backward_greedy_feature_selection(
        n_features_min=1, compute_error=False
    )

    print("BACKWARD DII TEST:")
    print("Weights with all features:\n", weights)
    print("Imbs with all features:\n", imbs)
    print("Feature Sets:", feature_sets)
    print("Final DIIs:", diis)

    # Check first set has all features
    assert set(feature_sets[0]) == {
        0,
        1,
        2,
    }, f"First feature set should have all features, got {feature_sets[0]}"

    # Check the feature that's removed first
    removed_feature = set(feature_sets[0]) - set(feature_sets[1])
    assert removed_feature == {
        expected_first_removal
    }, f"First removed feature should be {expected_first_removal}, got {removed_feature}"

    # Check second feature set
    assert set(feature_sets[1]) == set(
        expected_second_features
    ), f"Second feature set should be {expected_second_features}, got {feature_sets[1]}"

    # Check final feature set has only one feature
    assert (
        len(feature_sets[-1]) == 1
    ), f"Final feature set should have 1 feature, got {feature_sets[-1]}"


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_greedy_methods_shape():
    """Test that the greedy feature selection methods return the expected shapes.

    The dataset is a 3D Gaussian with variances [0.961146, 1.06219351, 0.01091285],
    where the third dimension has much lower variance. We apply weights [10, 0.1, 5]
    to the dimensions, making dimensions 0 and 2 more important.

    This test verifies that the greedy feature selection methods return outputs
    with the expected shapes and types.
    """
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data
    # The dataset has variances [0.961146, 1.06219351, 0.01091285]
    weights_ground_truth = np.array([1, 0.001, 5])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    # train the DII to recover ground-truth metric
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
        k_init=10,
        k_final=1,
        lambda_factor=1e-1,
        params_init=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        num_points_rows=None,
    )
    weights, imbs = dii.train()

    # Run forward greedy feature selection
    feature_sets_f, diis_f, errors_f = dii.forward_greedy_feature_selection(
        n_features_max=3, compute_error=False
    )

    # Run backward greedy feature selection
    feature_sets_b, diis_b, errors_b = dii.backward_greedy_feature_selection(
        n_features_min=1, compute_error=False
    )

    print("VERIFY SHAPES FORWARD:")
    print(feature_sets_f)
    print(diis_f)
    print(errors_f)

    print("VERIFY SHAPES BACKWARD:")
    print(feature_sets_b)
    print(diis_b)
    print(errors_b)

    # Check forward greedy results
    assert (
        len(feature_sets_f) == 3
    ), f"Forward selection should return 3 feature sets, got {len(feature_sets_f)}"
    assert (
        len(diis_f) == 3
    ), f"Forward selection should return 3 DII values, got {len(diis_f)}"
    assert errors_f == [
        None,
        None,
        None,
    ], f"Forward selection with compute_error=False should return None errors, got {errors_f}"

    # Check backward greedy results - expecting 3 feature sets based on previous test
    assert (
        len(feature_sets_b) == 3
    ), f"Backward selection should return 3 feature sets, got {len(feature_sets_b)}"
    assert (
        len(diis_b) == 3
    ), f"Backward selection should return 3 DII values, got {len(diis_b)}"
    # The shape of 'errors' returned by the backward greedy search should be (n_features-1)
    # This is due to the way the while loop is implemented
    assert errors_b == [
        None,
        None,
        None,
    ], f"Backward selection with compute_error=False should return None errors, got {errors_b}"


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_greedy_symmetry_5d_gaussian():
    """Test that forward and backward greedy selection are symmetric.

    Also test that the correct variables are selected from the prototypical 5d-gaussian example.

    This test verifies that the forward and backward greedy feature selection methods
    select features in the reverse order of each other, and that the DII values match
    when reversed.
    It also works on the prototypical 5D gaussian example to see if it selects the correct feature.
    """
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data with 5 dimensions
    np.random.seed(0)
    weights_ground_truth = np.array([10, 3, 1, 30, 7.3])
    data_A = np.random.normal(loc=0, scale=1.0, size=(500, 5))
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    # train the DII to recover ground-truth metric
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
        k_init=10,
        k_final=1,
        lambda_factor=1e-1,
        params_init=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        num_points_rows=None,
    )
    weights, imbs = dii.train()

    # Run forward and backward greedy feature selection
    feature_sets_fw, diis_fw, _ = dii.forward_greedy_feature_selection(
        n_features_max=5, compute_error=False
    )
    feature_sets_bw, diis_bw, _ = dii.backward_greedy_feature_selection(
        n_features_min=1, compute_error=False
    )

    # Expected results based on weights
    expected_fw_sets = [[3], [3, 0], [3, 0, 4], [3, 0, 4, 1], [3, 0, 4, 1, 2]]
    expected_bw_sets = [[0, 1, 2, 3, 4], [0, 1, 3, 4], [0, 3, 4], [0, 3], [3]]

    # Check forward greedy results
    assert (
        feature_sets_fw == expected_fw_sets
    ), f"Forward selection should return {expected_fw_sets}, got {feature_sets_fw}"

    # Check backward greedy results
    assert (
        feature_sets_bw == expected_bw_sets
    ), f"Backward selection should return {expected_bw_sets}, got {feature_sets_bw}"

    # Check that the DII values match when reversed
    diis_fw_array = np.array(diis_fw)
    diis_bw_array = np.array(diis_bw)

    print("Forward DIIs:", diis_fw)
    print("Backward DIIs:", diis_bw)

    assert np.allclose(
        diis_bw_array, diis_fw_array[::-1], atol=1e-2
    ), f"DII values should match when reversed, got {diis_bw_array} and {diis_fw_array[::-1]}"

    # Check that the feature sets are in reverse order
    for i in range(len(feature_sets_fw)):
        assert set(feature_sets_fw[i]) == set(
            feature_sets_bw[-(i + 1)]
        ), f"Feature sets should be in reverse order, got {feature_sets_fw[i]} and {feature_sets_bw[-(i+1)]}"


if __name__ == "__main__":
    test_DiffImbalance_forward_greedy()
    test_DiffImbalance_backward_greedy()
    test_DiffImbalance_greedy_methods_shape()
    test_DiffImbalance_greedy_symmetry_5d_gaussian()
