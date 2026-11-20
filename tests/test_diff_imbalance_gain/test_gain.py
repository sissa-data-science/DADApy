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

"""Module for testing the Imbalance Gain computed with the Differentiable Information Imbalance."""

import os

import numpy as np
import pytest
from jax import config

config.update("jax_platform_name", "cpu")
filename = os.path.join(os.path.split(__file__)[0], "../rosslers_xtoy_trajectory.npy")

EXPECTED_GAIN_X_TO_Y = 0.0968
EXPECTED_GAIN_Y_TO_X = 0.0023


def _spaces(tau=5, num_samples=200):
    """Present and future of two coupled Rossler systems, in which X (columns 0-2) drives Y (columns 3-5)."""
    traj = np.load(filename)
    traj = traj / traj.std(axis=0)
    t0s = np.linspace(0, len(traj) - tau - 1, num_samples, dtype=int)
    return traj[t0s, :3], traj[t0s, 3:], traj[t0s + tau, :3], traj[t0s + tau, 3:]


def test_gain_direction():
    """The gain is large from the driving to the driven system, and compatible with zero in the opposite direction."""
    from dadapy.diff_imbalance_gain import return_diff_imbalance_gain

    X0, Y0, Xtau, Ytau = _spaces()
    x_to_y = return_diff_imbalance_gain(X0, Y0, Ytau, k=5, num_epochs=200)
    y_to_x = return_diff_imbalance_gain(Y0, X0, Xtau, k=5, num_epochs=200)

    assert x_to_y["imbalance_gain"] == pytest.approx(EXPECTED_GAIN_X_TO_Y, abs=1e-3)
    assert y_to_x["imbalance_gain"] == pytest.approx(EXPECTED_GAIN_Y_TO_X, abs=1e-3)
    # weights with unit norm, cause first
    assert np.linalg.norm(x_to_y["optimized_weights"]) == pytest.approx(1)
    assert abs(x_to_y["optimized_weights"][0]) > abs(y_to_x["optimized_weights"][0])


def test_gain_against_diff_imbalance():
    """Baseline and optimized DII coincide with the DiffImbalance optimizations written explicitly."""
    from dadapy import DiffImbalance
    from dadapy.diff_imbalance_gain import return_diff_imbalance_gain

    X0, Y0, _, Ytau = _spaces(num_samples=122)
    periods_X, periods_Y = [0.0, 0.0, 4.0], [
        3.0,
        0.0,
        0.0,
    ]  # arbitrary periods, per column
    options = dict(
        k=4,
        num_epochs=8,
        batches_per_epoch=3,  # 122 points: the last 2 are discarded
        discard_close_ind=2,
        seed=1,
        num_epochs_average=3,
        lambda_factor=0.5,
        track_full_loss=True,
    )
    init = np.array([1.0, 2.0, 3.0, 4.0])

    result = return_diff_imbalance_gain(
        X0,
        Y0,
        Ytau,
        cause_params_groups=[2, 1],
        effect_params_groups=[1, 2],
        period_cause=[0, 4],  # per group
        period_effect=periods_Y,  # per column
        params_init=init,
        **options,
    )

    def train(data_A, **kwargs):
        dii = DiffImbalance(data_A=data_A, data_B=Ytau, periods_B=periods_Y, **kwargs)
        dii.train()
        return float(dii.return_final_dii()), np.asarray(dii.params_final)

    baseline, baseline_weights = train(
        Y0,
        periods_A=periods_Y,
        params_groups=[1, 2],
        params_init=init[2:] / np.linalg.norm(init[2:]),
        **options,
    )
    optimized, optimized_weights = train(
        np.concatenate((X0, Y0), axis=1),
        periods_A=periods_X + periods_Y,
        params_groups=[2, 1, 1, 2],
        params_init=init / np.linalg.norm(init),
        **options,
    )
    assert result["baseline_imbalance"] == pytest.approx(baseline, abs=1e-10)
    assert result["optimized_imbalance"] == pytest.approx(optimized, abs=1e-10)
    assert result["baseline_weights"] == pytest.approx(baseline_weights, abs=1e-10)
    assert result["optimized_weights"] == pytest.approx(optimized_weights, abs=1e-10)
    assert result["imbalance_gain"] == pytest.approx(
        (baseline - optimized) / baseline, abs=1e-10
    )
    assert result["optimized_params_groups"] == [2, 1, 1, 2]


def test_single_weight_baseline():
    """A baseline with a single weight is not trained and equals the joint space with zero weight on the cause."""
    from dadapy import DiffImbalance
    from dadapy.diff_imbalance_gain import return_diff_imbalance_gain

    X0, Y0, _, Ytau = _spaces(num_samples=100)
    result = return_diff_imbalance_gain(X0, Y0, Ytau, k=5, num_epochs=5)
    assert result["baseline_training_imbalances"].shape == (1,)
    assert result["baseline_training_weights"].shape == (1, 1)
    assert result["optimized_training_imbalances"].shape == (6,)
    assert result["optimized_training_weights"].shape == (6, 2)

    dii = DiffImbalance(
        data_A=np.concatenate((X0, Y0), axis=1),
        data_B=Ytau,
        params_groups=[3, 3],
        params_init=[0.0, 1.0],
        k=5,
        num_epochs=0,
    )
    dii.train()
    assert result["baseline_imbalance"] == pytest.approx(
        float(dii.return_final_dii()), abs=1e-10
    )


def test_class_against_function():
    """The DiffImbalanceGain class builds the spaces of the function, with time-delay embeddings and periods."""
    from dadapy.diff_imbalance_gain import (
        DiffImbalanceGain,
        return_diff_imbalance_gain,
    )

    traj = np.load(filename)
    periods = [0, 0, 50.0, 0, 30.0, 0]
    time_lags, num_samples, E, tau_e = [3, 8], 80, 2, 4
    options = dict(k=5, num_epochs=5, batches_per_epoch=2, learning_rate=0.02)

    dig = DiffImbalanceGain(
        time_series=traj, periods=periods, standardize=False, seed=2
    )
    results = dig.return_imbalance_gain(
        cause_variables=[0, 2],
        effect_variables=[4],
        time_lags=time_lags,
        num_samples=num_samples,
        embedding_dim=E,
        embedding_time=tau_e,
        cause_params_groups=[E, E],
        discard_close_ind=24,
        **options,
    )
    assert len(results) == len(time_lags) and dig.imbalance_gains is results

    # the same spaces built explicitly: samples every 'step' time steps, variable by variable
    t_first, t_last = (E - 1) * tau_e, len(traj) - max(time_lags) - 1
    step = (t_last - t_first) // (num_samples - 1)
    t0s = t_first + step * np.arange(num_samples)

    def embed(variables, times):
        return np.column_stack(
            [traj[times - i * tau_e, v] for v in variables for i in range(E)]
        )

    for tau, result in zip(time_lags, results):
        expected = return_diff_imbalance_gain(
            embed([0, 2], t0s),
            embed([4], t0s),
            embed([4], t0s + tau),
            cause_params_groups=[E, E],
            period_cause=[0, 0, 50.0, 50.0],
            period_effect=30.0,
            discard_close_ind=24 // step,
            seed=2,
            **options,
        )
        for key in ("baseline_imbalance", "optimized_imbalance", "optimized_weights"):
            assert result[key] == pytest.approx(expected[key], abs=1e-10)

    # a subset of the time lags on the same samples
    subset = dig.return_imbalance_gain(
        cause_variables=[0, 2],
        effect_variables=[4],
        time_lags=[3],
        time_lag_max=8,
        num_samples=num_samples,
        embedding_dim=E,
        embedding_time=tau_e,
        cause_params_groups=[E, E],
        discard_close_ind=24,
        **options,
    )
    assert subset[0]["optimized_imbalance"] == pytest.approx(
        results[0]["optimized_imbalance"], abs=1e-10
    )


def test_class_periods_and_standardization():
    """Periods are rescaled with the variables when these are standardized; variables can be given in any form."""
    from dadapy.diff_imbalance_gain import DiffImbalanceGain

    rng = np.random.default_rng(0)
    phases = np.cumsum(0.3 + 0.3 * rng.standard_normal((600, 3)), axis=0) % (2 * np.pi)
    series = np.column_stack(
        (phases, rng.standard_normal(600))
    )  # three angles and a nonperiodic variable
    periods = [2 * np.pi, 2 * np.pi, 2 * np.pi, 0]
    options = dict(time_lags=[2], num_samples=100, k=5, num_epochs=5)

    raw = DiffImbalanceGain(time_series=series, periods=periods, standardize=False)
    standardized = DiffImbalanceGain(time_series=series, periods=periods)
    assert standardized.periods == pytest.approx(periods / series.std(axis=0, ddof=1))

    # the baseline has a single variable: dividing it and its period by a constant changes nothing
    # (the joint space starts from equal weights on rescaled variables, so its training differs)
    reference = raw.return_imbalance_gain(
        [0, 3], [1], cause_params_groups=[1, 1], **options
    )[0]
    result = standardized.return_imbalance_gain(
        (0, 3), np.array([1]), cause_params_groups=[1, 1], **options
    )[0]
    assert result["baseline_imbalance"] == pytest.approx(
        reference["baseline_imbalance"], abs=1e-10
    )

    # without the periods the angles are not recognized as such
    no_periods = DiffImbalanceGain(time_series=series, standardize=False)
    different = no_periods.return_imbalance_gain(
        [0, 3], 1, cause_params_groups=[1, 1], **options
    )[0]
    assert abs(different["baseline_imbalance"] - reference["baseline_imbalance"]) > 1e-3


def test_class_ensemble():
    """With an ensemble of trajectories each trajectory is one sample, taken after the embedding window."""
    from dadapy.diff_imbalance_gain import (
        DiffImbalanceGain,
        return_diff_imbalance_gain,
    )

    traj = np.load(filename)
    ensemble = np.stack(
        [traj[i : i + 12] for i in range(0, 960, 12)]
    )  # 80 trajectories of 12 times
    options = dict(k=5, num_epochs=5)
    dig = DiffImbalanceGain(time_series_ensemble=ensemble, standardize=False, seed=3)
    result = dig.return_imbalance_gain(
        [0, 1], [3], time_lags=[4], embedding_dim=2, embedding_time=3, **options
    )[0]

    # t = 0 is the time index 3 of each trajectory; columns: variable by variable, then t and t - 3
    expected = return_diff_imbalance_gain(
        np.column_stack(
            (ensemble[:, 3, 0], ensemble[:, 0, 0], ensemble[:, 3, 1], ensemble[:, 0, 1])
        ),
        np.column_stack((ensemble[:, 3, 3], ensemble[:, 0, 3])),
        np.column_stack((ensemble[:, 7, 3], ensemble[:, 4, 3])),
        seed=3,
        **options,
    )
    assert result["baseline_imbalance"] == pytest.approx(
        expected["baseline_imbalance"], abs=1e-10
    )
    assert result["optimized_imbalance"] == pytest.approx(
        expected["optimized_imbalance"], abs=1e-10
    )


def test_wrong_inputs():
    """Inconsistent inputs raise a ValueError."""
    from dadapy.diff_imbalance_gain import (
        DiffImbalanceGain,
        return_diff_imbalance_gain,
    )

    X0, Y0, _, Ytau = _spaces(num_samples=60)
    wrong_arguments = [
        dict(cause_present=X0[:50]),  # different numbers of points
        dict(cause_params_groups=[1, 1]),  # groups not summing to the number of columns
        dict(params_init=[0.0, 1.0]),  # a weight initialized to zero
        dict(period_cause=[1.0, 2.0]),  # neither one period per column nor per group
        dict(period_effect=-1.0),
        dict(effect_future=np.where(Ytau > 1, np.nan, Ytau)),
    ]
    for arguments in wrong_arguments:
        with pytest.raises(ValueError):
            return_diff_imbalance_gain(
                **{
                    **dict(
                        cause_present=X0, effect_present=Y0, effect_future=Ytau, k=3
                    ),
                    **arguments,
                }
            )

    dig = DiffImbalanceGain(time_series=np.load(filename))
    wrong_arguments = [
        dict(cause_variables=[]),
        dict(cause_variables=[6]),  # only variables 0-5 exist
        dict(effect_variables=[0, 3]),  # shares a variable with the cause
        dict(time_lags=[0]),
        dict(time_lags=[5], time_lag_max=4),
        dict(embedding_dim=0),
    ]
    for arguments in wrong_arguments:
        with pytest.raises(ValueError):
            dig.return_imbalance_gain(
                **{
                    **dict(
                        cause_variables=[0],
                        effect_variables=[3],
                        time_lags=[5],
                        num_samples=50,
                        k=3,
                    ),
                    **arguments,
                }
            )
    with pytest.raises(ValueError):
        DiffImbalanceGain(time_series=np.load(filename), periods=-1.0)
