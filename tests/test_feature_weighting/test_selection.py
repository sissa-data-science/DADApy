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

"""Module for testing selection algorithms utilising differentiable information imbalance."""

import itertools

import numpy as np
import pytest

from dadapy import Data, FeatureWeighting

rng = np.random.default_rng(seed=0)


def test_optimise_imbalance_typing():
    """Test whether FeatureWeighting class methods throw errors for wrong input types."""
    data = rng.random((10, 5))

    for period in [
        [3],
        "faz",
        np.array([2, 2], dtype=np.int8),
        np.array([2, 2], dtype=np.float32),
    ]:
        feature_selection = FeatureWeighting(data, period=period)
        with pytest.raises(ValueError):
            feature_selection.return_weights_optimize_dii(Data(data), 1)

    for initial_weights in [np.array([2, 2], np.float32), "faz"]:
        feature_selection = FeatureWeighting(data)
        with pytest.raises(ValueError):
            feature_selection.return_weights_optimize_dii(
                Data(data), initial_weights=initial_weights
            )
    for lrs in [2, np.array([2, 2], np.float32), "faz"]:
        feature_selection = FeatureWeighting(data)
        with pytest.raises(ValueError):
            feature_selection.return_weights_optimize_dii(Data(data), decaying_lr=lrs)


def test_dist_matrix():
    """Test proper handling of class attribute full_distance_matrix.

    Attribute is set on first call and afterwards is not recalculated.
    Also should not accept wrongly shaped arrays.
    """
    data = rng.random((10, 5))
    feature_selection = FeatureWeighting(data)
    assert feature_selection._full_distance_matrix is None
    distance_matrix = feature_selection.full_distance_matrix
    assert feature_selection._full_distance_matrix is not None

    with pytest.raises(ValueError):
        feature_selection.full_distance_matrix = np.zeros((9, 8))

    assert distance_matrix.shape[0] == feature_selection.N
    assert distance_matrix.shape[1] == feature_selection.N


def test_maxk_warning():
    """Make sure FeatureWeighting properly warns when using maxk."""
    data = rng.random((10, 5))
    feature_selection = FeatureWeighting(data, maxk=3)

    with pytest.warns(UserWarning):
        feature_selection.return_dii_gradient(Data(data), weights=np.zeros(5))


def test_dii_gradient():
    """Test whether cython gradient matches much slower numpy gradient."""
    data = rng.random((10, 5))
    feature_selection = FeatureWeighting(data, maxk=3)

    cython_grad = feature_selection.return_dii_gradient(
        Data(data * rng.random(size=(5,))), weights=np.zeros(5)
    )
    feature_selection._cythond = False
    np_grad = feature_selection.return_dii_gradient(Data(data), weights=np.zeros(5))
    assert np.allclose(cython_grad, np_grad)


def test_optimise_imbalance():
    """Test dii optimisation on scaled random data.

    This should somewhat recapture the scaling values, or at least the relative importance.
    """
    data = rng.random((50, 5))
    weights_array = np.array([1, 1, 1e-2, 1e-2, 1e-2])
    target_data = data * weights_array
    periods = [None, np.ones(5)]
    constrains = [True, False]
    initial_weightss = [None, 1.0, weights_array]
    lambdas = [1e-5, 1, None]
    l1_penalties = [1.0, 10, 0.0]
    decays = ["cos", "exp", "static"]
    n_epochs = 5

    for (
        period,
        constrain,
        initial_weights,
        lambda_,
        l1_penalty,
        decay,
    ) in itertools.product(
        periods, constrains, initial_weightss, lambdas, l1_penalties, decays
    ):
        feature_selection = FeatureWeighting(data, period=period)
        assert feature_selection.history is None
        weights = feature_selection.return_weights_optimize_dii(
            Data(target_data),
            n_epochs=n_epochs,
            learning_rate=1e-2,
            constrain=constrain,
            initial_weights=initial_weights,
            lambd=lambda_,
            l1_penalty=l1_penalty,
            decaying_lr=decay,
        )
        assert weights.shape[0] == len(weights_array)
        assert feature_selection.history is not None

        assert isinstance(feature_selection.history["weights_per_epoch"], np.ndarray)
        assert feature_selection.history["weights_per_epoch"].shape[0] == n_epochs + 1
        assert feature_selection.history["weights_per_epoch"].shape[-1] == len(
            weights_array
        )

        assert isinstance(feature_selection.history["dii_per_epoch"], np.ndarray)
        assert feature_selection.history["dii_per_epoch"].shape[0] == n_epochs + 1
        assert isinstance(feature_selection.history["l1_term_per_epoch"], np.ndarray)
        assert feature_selection.history["l1_term_per_epoch"].shape[0] == n_epochs + 1

    data = rng.normal(size=(20, 5))
    weights_array = np.array([1, 1, 1e-3, 1e-3, 1e-3])
    target_data = data * weights_array
    feature_selection = FeatureWeighting(data, period=None)
    weights = feature_selection.return_weights_optimize_dii(
        Data(target_data),
        n_epochs=40,
        learning_rate=None,
        constrain=True,
        initial_weights=np.ones_like(weights_array),
        lambd=None,
        l1_penalty=1e-5,
        decaying_lr="exp",
    )
    assert np.all(weights[0] >= weights[2:])
    assert np.all(weights[1] >= weights[2:])


def test_optimal_learning_rate():
    """Test optimal learning rate and its dictionary history."""
    data = rng.random((20, 5))
    weights_array = np.array([1, 1, 1e-2, 1e-2, 1e-2])
    target_data = data * weights_array
    feature_selection = FeatureWeighting(data, period=None)

    trial_learning_rates = [1e-3, 1e-2]
    n_epochs = 5
    _ = feature_selection.return_optimal_learning_rate(
        target_data=Data(target_data),
        n_epochs=n_epochs,
        n_samples=4,
        trial_learning_rates=trial_learning_rates,
    )

    assert feature_selection.history is not None
    assert feature_selection.history["dii_per_epoch_per_lr"].shape[0] == len(
        trial_learning_rates
    )
    assert feature_selection.history["dii_per_epoch_per_lr"].shape[1] == n_epochs + 1
    assert feature_selection.history["weights_per_epoch_per_lr"].shape[0] == len(
        trial_learning_rates
    )
    assert (
        feature_selection.history["weights_per_epoch_per_lr"].shape[1] == n_epochs + 1
    )
    assert feature_selection.history["weights_per_epoch_per_lr"].shape[2] == len(
        weights_array
    )

    _ = feature_selection.return_optimal_learning_rate(
        target_data=Data(target_data),
        n_epochs=n_epochs,
        n_samples=500,
    )

    assert feature_selection.history is not None
    learning_rates = feature_selection.history["trial_learning_rates"]
    assert feature_selection.history["dii_per_epoch_per_lr"].shape[0] == len(
        learning_rates
    )
    assert feature_selection.history["dii_per_epoch_per_lr"].shape[1] == n_epochs + 1
    assert feature_selection.history["weights_per_epoch_per_lr"].shape[0] == len(
        learning_rates
    )
    assert (
        feature_selection.history["weights_per_epoch_per_lr"].shape[1] == n_epochs + 1
    )
    assert feature_selection.history["weights_per_epoch_per_lr"].shape[2] == len(
        weights_array
    )


def test_eliminate_backward_greedy_kernel_imbalance():
    """Test backward greedy and dictionary entries."""
    data = rng.random((20, 5))
    weights_array = np.array([1, 1, 1e-2, 1e-2, 1e-2])
    target_data = data * weights_array
    feature_selection = FeatureWeighting(data, period=None)

    n_epochs = 10
    _ = feature_selection.return_backward_greedy_dii_elimination(
        target_data=Data(target_data), n_epochs=n_epochs, constrain=False
    )

    assert feature_selection.history is not None
    assert feature_selection.history["dii_per_epoch"].shape[0] == len(weights_array)
    assert feature_selection.history["dii_per_epoch"].shape[1] == n_epochs + 1
    assert feature_selection.history["weights_per_epoch"].shape[0] == len(weights_array)
    assert feature_selection.history["weights_per_epoch"].shape[1] == n_epochs + 1
    assert feature_selection.history["weights_per_epoch"].shape[2] == len(weights_array)


def test_search_lasso_optimization_kernel_imbalance():
    """Test lasso optimization and dictionary entries.

    Here all available options are tested except for non-cythonized code,
    which works, but is too slow to be part of unittests.
    """
    data = rng.random((50, 5))
    weights_array = np.array([1, 1, 1e-2, 1e-2, 1e-2])
    target_data = data * weights_array
    feature_selection = FeatureWeighting(data, period=None)
    l1_penalties_options = [[1e-3, 1e-2, 1e-1], np.array([1e-5]), 1e-5, None]
    l1_decay_options = ["cos", "exp", "static"]

    n_epochs = 10
    for l1_penalties, constrain, refine, decaying_lr in itertools.product(
        l1_penalties_options, *([[True, False]] * 2), l1_decay_options
    ):
        (
            num_nonzero_features,
            l1_penalties_opt_per_nfeatures,
            dii_opt_per_nfeatures,
            weights_opt_per_nfeatures,
        ) = feature_selection.return_lasso_optimization_dii_search(
            target_data=Data(target_data),
            n_epochs=n_epochs,
            l1_penalties=l1_penalties,
            constrain=constrain,
            decaying_lr=decaying_lr,
            refine=refine,
            plotlasso=False,
        )

        assert feature_selection.history is not None
        assert num_nonzero_features.shape[0] == data.shape[1]
        assert dii_opt_per_nfeatures.shape[0] == data.shape[1]
        assert weights_opt_per_nfeatures.shape[0] == data.shape[1]
        assert num_nonzero_features.shape[0] == l1_penalties_opt_per_nfeatures.shape[0]
        assert dii_opt_per_nfeatures.shape[0] == l1_penalties_opt_per_nfeatures.shape[0]
        assert (
            weights_opt_per_nfeatures.shape[0]
            == l1_penalties_opt_per_nfeatures.shape[0]
        )
        assert weights_opt_per_nfeatures.shape[1] == target_data.shape[1]
