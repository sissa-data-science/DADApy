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
    feature_sets, diis, _, weights = dii.forward_greedy_feature_selection(
        n_features_max=3,
        compute_error=False,
        n_best=1,
    )

    print("FORWARD DII TEST:")
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
    feature_sets, diis, _, weights = dii.backward_greedy_feature_selection(
        n_features_min=1,
        compute_error=False,
        n_best=1,
    )

    print("BACKWARD DII TEST:")
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
def test_DiffImbalance_greedy_symmetry_5d_gaussian():
    """Test that forward and backward greedy selection are symmetric.

    Also test that the correct variables are selected from the prototypical 5d-gaussian example.

    This test verifies that the forward and backward greedy feature selection methods
    select features in the reverse order of each other, and that the DII values match
    when reversed.
    It also works on the prototypical 5D gaussian example to see if it selects the correct feature.
    This also checks if mini-batches and the error estimation works fine in the greedy implementation.
    """
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data with 5 dimensions
    np.random.seed(0)
    weights_ground_truth = np.array([10, 3, 1, 30, 7.3])
    data_A = np.random.normal(loc=0, scale=1.0, size=(100, 5))
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    # train the DII to recover ground-truth metric
    dii = DiffImbalance(
        data_A,
        data_B,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=2,
        l1_strength=0.0,
        point_adapt_lambda=True,
        k_init=3,
        k_final=3,
        lambda_factor=1e-1,
        params_init=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        num_points_rows=None,
    )
    weights, imbs = dii.train()

    # Run forward and backward greedy feature selection
    (
        feature_sets_fw,
        diis_fw,
        errors_fw,
        weights_fw,
    ) = dii.forward_greedy_feature_selection(
        n_features_max=5, compute_error=True, n_best=1
    )
    (
        feature_sets_bw,
        diis_bw,
        errors_bw,
        weights_bw,
    ) = dii.backward_greedy_feature_selection(
        n_features_min=1, compute_error=True, n_best=1
    )

    # Expected results based on weights
    expected_fw_sets = [[3], [0, 3], [0, 3, 4], [0, 1, 3, 4], [0, 1, 2, 3, 4]]
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
    print("Forward Errors:", errors_fw)
    print("Forward Weights:", weights_fw)
    print("Backward DIIs:", diis_bw)
    print("Backward Errors:", errors_bw)
    print("Backward Weights:", weights_bw)

    # Check that the weights are properly structured
    assert (
        len(weights_fw[-1]) == 5
    ), f"The final weight array should have length 5, got {len(weights_fw[-1])}"
    assert (
        len(weights_bw[-1]) == 5
    ), f"The final weight array should have length 5, got {len(weights_bw[-1])}"
    assert (
        len(weights_fw) == 5
    ), f"Forward selection should return 5 DII weights arrays, got {len(weights_fw)}"
    assert (
        len(weights_bw) == 5
    ), f"Backward selection should return 5 DII weights arrays, got {len(weights_bw)}"

    assert (
        len(errors_fw) == 5
    ), f"Forward selection should return 5 DII errors, got {len(errors_fw)}"
    assert (
        len(errors_bw) == 5
    ), f"Backward selection should return 5 DII errors, got {len(errors_bw)}"

    assert np.allclose(
        diis_bw_array, diis_fw_array[::-1], atol=1e-2
    ), f"DII values should match when reversed, got {diis_bw_array} and {diis_fw_array[::-1]}"

    # Check that for the forward selection, the last feature set has highest weight on feature 3
    # which should be the most important feature according to the ground truth weights
    max_weight_feature_fw = np.argmax(weights_fw[-1])
    assert (
        max_weight_feature_fw == 3
    ), f"Feature 3 should have the highest weight, got feature {max_weight_feature_fw}"

    # Check that for the backward selection, the first weights array (all features)
    # has highest weight on feature 3 as well
    max_weight_feature_bw = np.argmax(weights_bw[0])
    assert (
        max_weight_feature_bw == 3
    ), f"Feature 3 should have the highest weight, got feature {max_weight_feature_bw}"

    # Check that the feature sets are in reverse order
    for i in range(len(feature_sets_fw)):
        assert set(feature_sets_fw[i]) == set(
            feature_sets_bw[-(i + 1)]
        ), f"Feature sets should be in reverse order, got {feature_sets_fw[i]} and {feature_sets_bw[-(i + 1)]}"


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_greedy_random_initialization():
    """Test greedy feature selection with random initialization parameters.

    This test verifies that the greedy feature selection methods work correctly
    when initialized with random parameters between 0.1 and 5, ensuring that
    the initialization values are properly inherited from the parent class
    rather than using hardcoded values.
    """
    from dadapy import DiffImbalance  # noqa: E402

    # generate test data with 5 dimensions
    np.random.seed(0)
    weights_ground_truth = np.array([10, 3, 1, 30, 7.3])
    data_A = np.random.normal(loc=0, scale=1.0, size=(100, 5))
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    # Create random initialization parameters between 0.1 and 5
    np.random.seed(42)  # Different seed for initialization
    params_init = np.random.uniform(0.1, 1.0, size=5)
    print(f"Random initialization parameters: {params_init}")

    # train the DII to recover ground-truth metric with random initialization
    dii = DiffImbalance(
        data_A,
        data_B,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=2,
        l1_strength=0.0,
        point_adapt_lambda=True,
        k_init=3,
        k_final=3,
        lambda_factor=1e-1,
        params_init=params_init,  # Use random initialization
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        num_points_rows=None,
    )
    weights, imbs = dii.train()

    # Run forward and backward greedy feature selection
    (
        feature_sets_fw,
        diis_fw,
        errors_fw,
        weights_fw,
    ) = dii.forward_greedy_feature_selection(
        n_features_max=5, compute_error=True, n_best=1
    )
    (
        feature_sets_bw,
        diis_bw,
        errors_bw,
        weights_bw,
    ) = dii.backward_greedy_feature_selection(
        n_features_min=1, compute_error=True, n_best=1
    )

    # Expected results based on weights (should be same as previous test)
    expected_fw_sets = [[3], [0, 3], [0, 3, 4], [0, 1, 3, 4], [0, 1, 2, 3, 4]]
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

    print("Random Init Forward DIIs:", diis_fw)
    print("Random Init Forward Errors:", errors_fw)
    print("Random Init Forward Weights:", weights_fw)
    print("Random Init Backward DIIs:", diis_bw)
    print("Random Init Backward Errors:", errors_bw)
    print("Random Init Backward Weights:", weights_bw)

    # Check that the weights are properly structured
    assert (
        len(weights_fw[-1]) == 5
    ), f"The final weight array should have length 5, got {len(weights_fw[-1])}"
    assert (
        len(weights_bw[-1]) == 5
    ), f"The final weight array should have length 5, got {len(weights_bw[-1])}"
    assert (
        len(weights_fw) == 5
    ), f"Forward selection should return 5 DII weights arrays, got {len(weights_fw)}"
    assert (
        len(weights_bw) == 5
    ), f"Backward selection should return 5 DII weights arrays, got {len(weights_bw)}"

    assert (
        len(errors_fw) == 5
    ), f"Forward selection should return 5 DII errors, got {len(errors_fw)}"
    assert (
        len(errors_bw) == 5
    ), f"Backward selection should return 5 DII errors, got {len(errors_bw)}"

    assert np.allclose(
        diis_bw_array, diis_fw_array[::-1], atol=1e-2
    ), f"DII values should match when reversed, got {diis_bw_array} and {diis_fw_array[::-1]}"

    # Check that for the forward selection, the last feature set has highest weight on feature 3
    # which should be the most important feature according to the ground truth weights
    max_weight_feature_fw = np.argmax(weights_fw[-1])
    assert (
        max_weight_feature_fw == 3
    ), f"Feature 3 should have the highest weight, got feature {max_weight_feature_fw}"

    # Check that for the backward selection, the first weights array (all features)
    # has highest weight on feature 3 as well
    max_weight_feature_bw = np.argmax(weights_bw[0])
    assert (
        max_weight_feature_bw == 3
    ), f"Feature 3 should have the highest weight, got feature {max_weight_feature_bw}"

    # Check that the feature sets are in reverse order
    for i in range(len(feature_sets_fw)):
        assert set(feature_sets_fw[i]) == set(
            feature_sets_bw[-(i + 1)]
        ), f"Feature sets should be in reverse order, got {feature_sets_fw[i]} and {feature_sets_bw[-(i + 1)]}"

    # Additional test: Verify that the random initialization was actually used
    # by checking that the initial DII object has the correct params_init
    assert np.allclose(
        dii.params_init, params_init
    ), f"DII object should have the random initialization parameters, got {dii.params_init} expected {params_init}"


if __name__ == "__main__":
    test_DiffImbalance_forward_greedy()
    test_DiffImbalance_backward_greedy()
    test_DiffImbalance_greedy_symmetry_5d_gaussian()
    test_DiffImbalance_greedy_random_initialization()
