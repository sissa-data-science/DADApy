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

"""Module for testing methods of the CausalGraph class."""

import os
import sys

import numpy as np
import pytest
from jax import config

config.update("jax_platform_name", "cpu")
filename = os.path.join(os.path.split(__file__)[0], "../rosslers_xtoy_trajectory.npy")


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_CausalGraph_optimization():
    """Test CausalGraph optimization function."""
    from dadapy import CausalGraph  # noqa: E402

    # generate test data
    traj = np.load(filename)

    expected_weights = [0.4423, 0.46, 0.3371, 0.4362, 0.3505, 0.3866]
    expected_imbs_last_epoch = [0.3189, 0.3706, 0.4485, 0.3384, 0.3644, 0.5043]

    # train the DII
    num_samples = 50
    num_epochs = 10
    batches_per_epoch = 1
    k = 5
    time_lags = [5]  # time lags tested

    g = CausalGraph(time_series=traj, seed=0)  # object of the class CausalGraph

    weights_final, imbs_training, _ = g.optimize_present_to_future(
        time_lags=time_lags,
        num_samples=num_samples,
        target_variables="all",
        save_weights=False,
        embedding_dim_present=1,
        embedding_dim_future=1,
        embedding_time=1,
        num_epochs=num_epochs,
        batches_per_epoch=batches_per_epoch,
        l1_strength=0.0,
        point_adapt_lambda=True,
        k=k,
        lambda_factor=0.1,
        params_init=None,
        optimizer_name="adam",
        learning_rate=1e-2,
        learning_rate_decay="cos",
        compute_imb_final=False,
        discard_close_ind=0,
    )

    # the DII is invariant under a sign change of each weight
    assert np.abs(weights_final[:, 0, 0]) == pytest.approx(expected_weights, abs=0.01)
    assert imbs_training[:, 0, -1] == pytest.approx(expected_imbs_last_epoch, abs=0.01)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_CausalGraph_periods_standardization():
    """Test that the periods are rescaled together with the variables when these are standardized."""
    from dadapy import CausalGraph  # noqa: E402

    # two coupled phases in degrees (the second follows the first) and a nonperiodic variable
    rng = np.random.default_rng(0)
    n_times = 2000
    series = np.zeros((n_times, 3))
    for t in range(1, n_times):
        series[t, 0] = series[t - 1, 0] + 20 + 10 * rng.standard_normal()
        series[t, 1] = (
            series[t - 1, 1]
            + 20
            + 30 * np.sin(np.deg2rad(series[t - 1, 0] - series[t - 1, 1]))
            + 5 * rng.standard_normal()
        )
        series[t, 2] = 0.8 * series[t - 1, 2] + rng.standard_normal()
    series[:, :2] = np.mod(series[:, :2], 360.0)
    periods = np.array([360.0, 360.0, 0.0])
    std = np.std(series, ddof=1, axis=0)

    kwargs = dict(
        time_lags=[1],
        num_samples=200,
        target_variables="all",
        save_weights=False,
        num_epochs=10,
        k=5,
    )
    # variables standardized by the class: the periods must follow
    g = CausalGraph(time_series=series, periods=periods, seed=0)
    assert g.periods == pytest.approx(periods / std)
    weights, imbs, _ = g.optimize_present_to_future(**kwargs)

    # same variables and periods, standardized by hand
    g_ref = CausalGraph(
        time_series=series / std, periods=periods / std, standardize=False, seed=0
    )
    weights_ref, imbs_ref, _ = g_ref.optimize_present_to_future(**kwargs)

    assert weights == pytest.approx(weights_ref, abs=1e-6)
    assert imbs == pytest.approx(imbs_ref, abs=1e-6)

    # the input periods are not modified, and the periods are not rescaled without standardization
    assert periods == pytest.approx([360.0, 360.0, 0.0])
    with pytest.warns(UserWarning):
        g_raw = CausalGraph(time_series=series, periods=periods, standardize=False)
    assert g_raw.periods == pytest.approx(periods)
