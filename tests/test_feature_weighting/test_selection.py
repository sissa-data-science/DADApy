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

rng = np.random.default_rng()


def test_optimise_imbalance_typing():
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

    for initial_gammas in [np.array([2, 2], np.float32), "faz"]:
        feature_selection = FeatureWeighting(data)
        with pytest.raises(ValueError):
            feature_selection.return_weights_optimize_dii(
                Data(data), initial_gammas=initial_gammas
            )


def test_dist_matrix():
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
    data = rng.random((10, 5))
    feature_selection = FeatureWeighting(data, maxk=3)

    with pytest.warns():
        feature_selection.return_dii_gradient(
            Data(data), gammas=np.zeros(5)
        )


def test_optimise_imbalance():
    data = rng.random((3, 5))
    weights_array = np.array([1, 1, 1e-2, 1e-2, 1e-2])
    target_data = data * weights_array
    periods = [None, np.ones(5)]
    constrains = [True, False]
    initial_gammass = [None, 1.0, weights_array]
    lambdas = [1e-5, 1, None]
    l1_penalties = [1.0, 10, 0.0]
    decays = [True, False]
    n_epochs = 2

    for (
        period,
        constrain,
        initial_gammas,
        lambda_,
        l1_penalty,
        decay,
    ) in itertools.product(
        periods, constrains, initial_gammass, lambdas, l1_penalties, decays
    ):
        feature_selection = FeatureWeighting(data, period=period)
        assert feature_selection.history is None
        gammas = feature_selection.return_weights_optimize_dii(
            Data(target_data),
            n_epochs=n_epochs,
            learning_rate=1e-2,
            constrain=constrain,
            initial_gammas=initial_gammas,
            lambd=lambda_,
            l1_penalty=l1_penalty,
            decaying_lr=decay,
        )
        assert gammas.shape[0] == len(weights_array)
        assert feature_selection.history is not None

        assert isinstance(feature_selection.history["weights_per_epoch"], np.ndarray)
        assert feature_selection.history["weights_per_epoch"].shape[0] == n_epochs + 1
        assert feature_selection.history["weights_per_epoch"].shape[-1] == len(
            weights_array
        )

        assert isinstance(feature_selection.history["dii_per_epoch"], np.ndarray)
        assert feature_selection.history["dii_per_epoch"].shape[0] == n_epochs + 1
        assert isinstance(feature_selection.history["l1_term_per_epoch"], np.ndarray)
        assert (
            feature_selection.history["l1_term_per_epoch"].shape[0] == n_epochs + 1
        )

    data = rng.normal(size=(20, 5))
    weights_array = np.array([1, 1, 1e-2, 1e-2, 1e-2])
    target_data = data * weights_array
    feature_selection = FeatureWeighting(data, period=None)
    gammas = feature_selection.return_weights_optimize_dii(
        Data(target_data),
        n_epochs=30,
        learning_rate=None,
        constrain=True,
        initial_gammas=np.ones_like(weights_array),
        lambd=None,
        l1_penalty=1e-3,
        decaying_lr=decay,
    )
    assert (np.sum(gammas[0]) >= np.sum(gammas[2:]))
    assert (np.sum(gammas[1]) >= np.sum(gammas[2:]))


def test_optimal_learning_rate():
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
    assert feature_selection.history["gammas_per_epoch_per_lr"].shape[0] == len(
        trial_learning_rates
    )
    assert feature_selection.history["gammas_per_epoch_per_lr"].shape[1] == n_epochs + 1
    assert feature_selection.history["gammas_per_epoch_per_lr"].shape[2] == len(
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
    assert feature_selection.history["gammas_per_epoch_per_lr"].shape[0] == len(
        learning_rates
    )
    assert feature_selection.history["gammas_per_epoch_per_lr"].shape[1] == n_epochs + 1
    assert feature_selection.history["gammas_per_epoch_per_lr"].shape[2] == len(
        weights_array
    )


def test_eliminate_backward_greedy_kernel_imbalance():
    data = rng.random((20, 5))
    weights_array = np.array([1, 1, 1e-2, 1e-2, 1e-2])
    target_data = data * weights_array
    feature_selection = FeatureWeighting(data, period=None)

    n_epochs = 10
    _ = feature_selection.return_backward_greedy_dii_elimination(
        target_data=Data(target_data), n_epochs=n_epochs, constrain=False
    )

    assert feature_selection.history is not None
    assert (
        feature_selection.history["dii_per_epoch"].shape[0] == len(weights_array)
    )
    assert feature_selection.history["dii_per_epoch"].shape[1] == n_epochs + 1


def test_search_lasso_optimization_kernel_imbalance():
    # TODO: Make this properly

    data = rng.random((20, 5))
    weights_array = np.array([1, 1, 1e-2, 1e-2, 1e-2])
    target_data = data * weights_array
    feature_selection = FeatureWeighting(data, period=None)

    n_epochs = 10
    (
        gammas_list,
        kernel_list,
        lassoterm_list,
        penalties,
    ) = feature_selection.return_lasso_optimization_dii_search(
        target_data=Data(target_data), n_epochs=n_epochs, constrain=False
    )
