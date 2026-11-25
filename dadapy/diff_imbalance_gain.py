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

"""
The *diff_imbalance_gain* module contains the *DiffImbalanceGain* class.

The Imbalance Gain (Del Tatto et al., PNAS 2024) quantifies how much the present of a putative cause X improves
the prediction of the future of an effect Y, with respect to the present of Y alone. Here it is computed with the
Differentiable Information Imbalance (DII) of the *DiffImbalance* class (Wild et al., Nat. Commun. 2025)::

    baseline  = DII( Y(0)         -> Y(tau) )
    optimized = DII( (X(0), Y(0)) -> Y(tau) )    with the weights of X(0) and Y(0) optimized
    gain      = (baseline - optimized) / baseline

The function *return_diff_imbalance_gain* computes the gain from the three spaces X(0), Y(0) and Y(tau); the
*DiffImbalanceGain* class builds these spaces from a time series, with the same conventions of the *CausalGraph*
class.

By default, JAX runs the code on GPU when one is available. A specific platform can be selected, before any
JAX computation, with the command
    jax.config.update('jax_platforms', 'cpu')  # set 'cpu' or 'cuda'
or with the environment variable JAX_PLATFORMS=cpu.
"""

import numpy as np

from dadapy.causal_graph import CausalGraph
from dadapy.diff_imbalance import DiffImbalance


def return_diff_imbalance_gain(
    cause_present,
    effect_present,
    effect_future,
    k=1,
    discard_close_ind=0,
    num_epochs=200,
    params_init=None,
    cause_params_groups=None,
    effect_params_groups=None,
    period_cause=None,
    period_effect=None,
    period_cause_groups=None,
    period_effect_groups=None,
    batches_per_epoch=1,
    track_full_loss=False,
    seed=0,
    l1_strength=0.0,
    point_adapt_lambda=True,
    lambda_factor=0.1,
    optimizer_name="adam",
    learning_rate=1e-2,
    learning_rate_decay=None,
    learning_rate_final=None,
    num_epochs_average=1,
):
    """Compute the Imbalance Gain from a cause X to an effect Y with the Differentiable Information Imbalance.

    The baseline is the DII from Y(0) to Y(tau), and the optimized value is the DII from the joint space
    (X(0), Y(0)) to Y(tau), where one weight is optimized for each group of columns of X(0) and Y(0). The space
    Y(tau) is never weighted. If Y(0) has a single weight, the baseline is evaluated without training, as a global
    weight does not change the distance ranks; if it has several weights, the baseline is trained with the same
    settings of the joint space, so that the gain is not inflated by a mere re-weighting of the effect variables.

    The DII only depends on the direction of the weights. The initial weights are rescaled to unit norm, so that
    the learning rate has the same meaning for any 'params_init' and the weights of different optimizations can be
    compared directly; only their absolute values are meaningful. The gain can be slightly negative when the cause
    carries no information, as the optimized DII is not constrained to be lower than the baseline.

    The k neighbors of the DII are weighted, so k is not directly the size of the neighborhood. The effective
    number of neighbors of a point, 1 / sum_j c_ij^2 for its DII coefficients c_ij, is about 0.64 k with
    lambda_factor=0.3, the value for which the baseline and the joint space have the same effective neighborhood.
    To compare with an Imbalance Gain computed with the standard Information Imbalance and k_std neighbors, use
    lambda_factor=0.3 and about 1.6 k_std neighbors for the final DII (k * batches_per_epoch).

    As a function of the weights the DII can have several local minima, so the final weights may depend on
    'params_init' and, with mini-batches, on 'seed': this should be checked on the data. A few mini-batches (e.g.
    batches_per_epoch=4) reduce the dependence on the initial weights; larger neighborhoods reduce it further but
    also reduce the gain. Since the weights are optimized and scored on the same points, a cause carrying no
    information still yields a small positive gain (a few tenths of a percent on time series of 1000-1600
    points), which can be estimated by shuffling the cause among the points.

    Args:
        cause_present (np.ndarray(float)): array of shape (n_points, n_cause_columns) or (n_points,), space X(0).
        effect_present (np.ndarray(float)): array of shape (n_points, n_effect_columns) or (n_points,), space Y(0).
        effect_future (np.ndarray(float)): array with the shape of effect_present, space Y(tau).
        k (int): distance rank of neighbors used to set the smoothing parameter lambda; with minibatches it
            refers to the points of one mini-batch. Default is 1.
        discard_close_ind (int): the pairs of points (i, j) with |i - j| <= discard_close_ind are discarded, both
            during the training and in the final DII, in the baseline and in the joint space. Default is 0.
        num_epochs (int): number of training epochs. Default is 200.
        params_init (np.ndarray(float)): array of shape (n_cause_groups + n_effect_groups,) containing the initial
            weights of the joint space, cause groups first. It is rescaled to unit norm and cannot contain zeros, as
            a weight initialized to zero is not updated. Default is None, for which the weights are all equal.
        cause_params_groups (list(int)): number of consecutive columns of X(0) sharing each weight, for example
            [E, E, E] for three variables with E time-delay components each. Default is None, for which all the
            columns of X(0) share a single weight.
        effect_params_groups (list(int)): same as cause_params_groups, for Y(0).
        period_cause (float, np.ndarray(float)): periods of the columns of X(0), with 0 or None for the nonperiodic
            ones. A scalar is applied to all the columns; an array can contain one period per column, or one period
            per group (of cause_params_groups, or of period_cause_groups if given). Default is None (nonperiodic).
        period_effect (float, np.ndarray(float)): same as period_cause, for the columns of Y(0) and Y(tau).
        period_cause_groups (list(int)): number of consecutive columns of X(0) sharing each entry of period_cause,
            when the periods do not follow the weight groups (for example a single weight for a space that mixes
            periodic and nonperiodic variables). Default is None, for which cause_params_groups is used.
        period_effect_groups (list(int)): same as period_cause_groups, for period_effect.
        batches_per_epoch (int): number of minibatches; see the DiffImbalance class. Default is 1.
        track_full_loss (bool): whether to compute the training DII on the full data set at each training epoch, if
            minibatches are used. Default is False.
        seed (int): seed of the JAX random generator, which determines the mini-batch partitions. Default is 0.
        l1_strength (float): strength of the L1 regularization, applied to the baseline (if trained) and to the
            joint space; see the DiffImbalance class. Default is 0.
        point_adapt_lambda (bool): whether to use a different smoothing parameter lambda for each point.
            Default is True.
        lambda_factor (float): factor defining the scale of lambda. Default is 0.1.
        optimizer_name (str): name of the optimizer, 'adam' (default) or 'sgd'.
        learning_rate (float): value of the learning rate. Default is 1e-2.
        learning_rate_decay (str): schedule to damp the learning rate: cosine decay ("cos") or constant learning
            rate (None). Default is None.
        learning_rate_final (float): final value of the learning rate with the "cos" schedule. Default is None, for
            which the learning rate is damped to zero.
        num_epochs_average (int): number of final training epochs over which the weights are averaged to define the
            final weights. Default is 1.

    Returns:
        result (dict): dictionary with keys
            'baseline_imbalance', 'optimized_imbalance', 'imbalance_gain' (float);
            'baseline_weights', 'optimized_weights' (np.ndarray(float)): final weights with unit norm, one per
            group, with the cause groups first in the joint space;
            'baseline_params_groups', 'optimized_params_groups' (list(int)): groups of columns of the weights;
            'baseline_training_imbalances', 'optimized_training_imbalances' (np.ndarray(float)): DII during the
            training, of shape (num_epochs+1,); it is computed on the full data set without minibatches or with
            track_full_loss=True, and on one mini-batch otherwise;
            'baseline_training_weights', 'optimized_training_weights' (np.ndarray(float)): weights during the
            training, of shape (num_epochs+1, n_groups).
            A baseline that is not trained has a single stored epoch: shapes (1,) and (1, 1).
    """
    cause_present = _as_2d(cause_present)
    effect_present = _as_2d(effect_present)
    effect_future = _as_2d(effect_future)
    n_cause, n_effect = cause_present.shape[1], effect_present.shape[1]
    if not len(cause_present) == len(effect_present) == len(effect_future):
        raise ValueError(
            "cause_present, effect_present and effect_future must have the same number of points."
        )
    if effect_future.shape[1] != n_effect:
        raise ValueError(
            "effect_present and effect_future must have the same number of columns."
        )

    cause_groups = _check_groups(cause_params_groups, n_cause, "cause_params_groups")
    effect_groups = _check_groups(
        effect_params_groups, n_effect, "effect_params_groups"
    )
    joint_groups = cause_groups + effect_groups

    period_cause = _expand_periods(
        period_cause,
        _check_groups(
            period_cause_groups, n_cause, "period_cause_groups", cause_groups
        ),
        "period_cause",
    )
    period_effect = _expand_periods(
        period_effect,
        _check_groups(
            period_effect_groups, n_effect, "period_effect_groups", effect_groups
        ),
        "period_effect",
    )
    periodic = period_cause.any() or period_effect.any()
    period_joint = np.concatenate((period_cause, period_effect)) if periodic else None
    period_effect = period_effect if periodic else None

    joint_init = None
    effect_init = None
    if params_init is not None:
        joint_init = np.asarray(params_init, dtype=float).ravel()
        if joint_init.size != len(joint_groups):
            raise ValueError(
                f"params_init must contain {len(joint_groups)} weights (one per group, cause groups first), "
                + f"while it contains {joint_init.size}."
            )
        if not np.all(np.isfinite(joint_init)) or np.any(joint_init == 0):
            raise ValueError(
                "params_init must contain finite, non-zero weights: a weight that starts at zero stays at zero."
            )
        effect_init = joint_init[len(cause_groups) :]

    dii_kwargs = dict(
        k=k,
        discard_close_ind=discard_close_ind,
        batches_per_epoch=batches_per_epoch,
        track_full_loss=track_full_loss,
        seed=seed,
        l1_strength=l1_strength,
        point_adapt_lambda=point_adapt_lambda,
        lambda_factor=lambda_factor,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        learning_rate_final=learning_rate_final,
    )
    # the baseline is trained only if it has more than one weight
    train_baseline = len(effect_groups) > 1
    baseline = _train_diff_imbalance(
        data_A=effect_present,
        data_B=effect_future,
        periods_A=period_effect,
        periods_B=period_effect,
        params_groups=effect_groups,
        params_init=effect_init,
        num_epochs=num_epochs if train_baseline else 0,
        num_epochs_average=num_epochs_average if train_baseline else 1,
        bar_label="baseline",
        **dii_kwargs,
    )
    optimized = _train_diff_imbalance(
        data_A=np.concatenate((cause_present, effect_present), axis=1),
        data_B=effect_future,
        periods_A=period_joint,
        periods_B=period_effect,
        params_groups=joint_groups,
        params_init=joint_init,
        num_epochs=num_epochs,
        num_epochs_average=num_epochs_average,
        bar_label="joint",
        **dii_kwargs,
    )

    imbalance_gain = np.nan
    if baseline["imbalance"] != 0:
        imbalance_gain = (baseline["imbalance"] - optimized["imbalance"]) / baseline[
            "imbalance"
        ]
    return {
        "baseline_imbalance": baseline["imbalance"],
        "optimized_imbalance": optimized["imbalance"],
        "imbalance_gain": imbalance_gain,
        "baseline_weights": baseline["weights"],
        "optimized_weights": optimized["weights"],
        "baseline_params_groups": effect_groups,
        "optimized_params_groups": joint_groups,
        "baseline_training_imbalances": baseline["training_imbalances"],
        "optimized_training_imbalances": optimized["training_imbalances"],
        "baseline_training_weights": baseline["training_weights"],
        "optimized_training_weights": optimized["training_weights"],
    }


def _train_diff_imbalance(data_A, data_B, params_init, bar_label, **dii_kwargs):
    """Train a DiffImbalance object, with initial weights of unit norm, and return its final DII and weights."""
    if params_init is not None:
        params_init = params_init / np.linalg.norm(params_init)
    dii = DiffImbalance(
        data_A=data_A, data_B=data_B, params_init=params_init, **dii_kwargs
    )
    training_weights, training_imbalances = dii.train(bar_label=bar_label)
    return {
        "imbalance": float(dii.return_final_dii()),
        "weights": np.asarray(dii.params_final),
        "training_imbalances": training_imbalances,
        "training_weights": training_weights,
    }


def _as_2d(data):
    """Return the data as an array of floats of shape (n_points, n_columns)."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {data.shape}.")
    if not np.all(np.isfinite(data)):
        raise ValueError("The data contain nan or infinite values.")
    return data


def _check_groups(groups, n_columns, name, default=None):
    """Return the groups as a list of positive integers summing to the number of columns."""
    if groups is None:
        return [n_columns] if default is None else default
    groups = np.ravel(groups)
    if (
        not np.issubdtype(groups.dtype, np.integer)
        or groups.min() <= 0
        or groups.sum() != n_columns
    ):
        raise ValueError(
            f"{name} must contain positive integers summing to {n_columns}, got {groups.tolist()}."
        )
    return groups.tolist()


def _expand_periods(periods, groups, name):
    """Return one period per column (0 if nonperiodic) from a scalar, per-group or per-column input."""
    n_columns = sum(groups)
    if periods is None:
        return np.zeros(n_columns)
    # entries equal to None become nan, then 0
    periods = np.nan_to_num(np.asarray(periods, dtype=float).ravel())
    if np.any(periods < 0):
        raise ValueError(f"{name} must not contain negative periods.")
    if periods.size == 1:
        return np.full(n_columns, periods[0])
    if periods.size == n_columns:
        return periods
    if periods.size == len(groups):
        return np.repeat(periods, groups)
    raise ValueError(
        f"{name} must be a scalar, or have one entry per group ({len(groups)}) or per column ({n_columns}); "
        + f"got {periods.size} entries."
    )


class DiffImbalanceGain(CausalGraph):
    """Computes the Imbalance Gain between groups of variables of a time series, using the DII.

    The data, the periods of the variables and the seed are provided as in the CausalGraph class, from which the
    selection of the samples and the construction of the time-delay embeddings are inherited.

    Attributes:
        time_series (np.array(float)): array of shape (N_times,D), where N_times is the length of
            trajectory and D is the number of dynamical variables.
        time_series_ensemble (np.array(float)): array of shape (N_trajs,N_times,D) containing N_trajs
            independent trajectories of the same dynamical process; read only when time_series is None.
        periods (np.ndarray(float)): array of shape (D,) containing the periods of the dynamical variables,
            with 0 for the nonperiodic ones, in the units of the input data. Default is None (all variables
            nonperiodic). If standardize is True, the periods are divided by the standard deviations together
            with the variables.
        standardize (bool): whether to divide each variable by its standard deviation. Default is True.
        seed (int): seed of JAX random generator.
        imbalance_gains (list(dict)): output of the last call to 'return_imbalance_gain'.
    """

    def __init__(
        self,
        time_series=None,
        time_series_ensemble=None,
        periods=None,
        standardize=True,
        seed=0,
    ):
        """Initialise the DiffImbalanceGain object."""
        super().__init__(
            time_series=time_series,
            time_series_ensemble=time_series_ensemble,
            periods=periods,
            standardize=standardize,
            seed=seed,
        )
        if periods is not None:
            periods = np.ones(self.num_variables) * np.array(periods)
            if np.any(periods < 0):
                raise ValueError("'periods' must not contain negative values.")
            if standardize:
                # the variables were divided by their standard deviations: the periods follow. They are
                # computed here from the input periods, whether or not CausalGraph rescales them
                data = time_series if time_series is not None else time_series_ensemble
                periods = periods / np.std(
                    np.reshape(data, (-1, self.num_variables)), ddof=1, axis=0
                )
            self.periods = periods
        self.imbalance_gains = None

    def return_imbalance_gain(
        self,
        cause_variables,
        effect_variables,
        time_lags,
        num_samples=None,
        embedding_dim=1,
        embedding_time=1,
        cause_params_groups=None,
        effect_params_groups=None,
        time_lag_max=None,
        num_epochs=200,
        batches_per_epoch=1,
        track_full_loss=False,
        l1_strength=0.0,
        point_adapt_lambda=True,
        k=1,
        lambda_factor=0.1,
        params_init=None,
        optimizer_name="adam",
        learning_rate=1e-2,
        learning_rate_decay=None,
        learning_rate_final=None,
        discard_close_ind=0,
        num_epochs_average=1,
    ):
        """Compute the Imbalance Gain from the cause variables to the effect variables, for each time lag.

        The spaces X(0), Y(0) and Y(tau) are built from the same samples for all the time lags. With time-delay
        embeddings, each variable is replaced by its values at times t, t - embedding_time, ...,
        t - (embedding_dim-1) * embedding_time, in adjacent columns; time lags shorter than this window make Y(0)
        and Y(tau) overlap, and should be avoided. The periods of the columns are taken from the attribute
        'periods'. See the function 'return_diff_imbalance_gain' for the definition of the gain.

        Args:
            cause_variables (list(int)): indices of the variables defining the putative cause X.
            effect_variables (list(int)): indices of the variables defining the effect Y, different from the
                cause variables.
            time_lags (list(int), np.ndarray(int)): tested time lags tau between 'present' and 'future'.
            num_samples (int): number of samples harvested from the full time series at uniformly spaced times,
                interpreted as independent initial conditions of the same dynamical process. Default is None, for
                which all the available times are used (with a warning). Ignored if the data are provided through
                'time_series_ensemble'.
            embedding_dim (int): dimension of the time-delay embedding vectors of each variable, in X(0), Y(0) and
                Y(tau). Default is 1, which means the time-delay embeddings are not employed.
            embedding_time (int): lag between consecutive samples in the time-delay embedding vectors. Default is 1.
            cause_params_groups (list(int)): number of consecutive columns of X(0) sharing each weight, where the
                columns are ordered by variable and then by time-delay component; for example
                [embedding_dim] * len(cause_variables) optimizes one weight per variable. Default is None, for
                which all the columns of X(0) share a single weight.
            effect_params_groups (list(int)): same as cause_params_groups, for Y(0).
            time_lag_max (int): largest time lag for which the samples are selected. Default is None, for which
                it is the largest of time_lags; set it to compute a subset of the time lags on the same samples.
            num_epochs (int): number of training epochs. Default is 200.
            batches_per_epoch (int): number of minibatches; see the DiffImbalance class. Default is 1.
            track_full_loss (bool): whether to compute the training DII on the full data set at each training
                epoch, if minibatches are used. Default is False.
            l1_strength (float): strength of the L1 regularization; see the DiffImbalance class. Default is 0.
            point_adapt_lambda (bool): whether to use a different smoothing parameter lambda for each point.
                Default is True.
            k (int): distance rank of neighbors used to set lambda; with minibatches it refers to the points of
                one mini-batch. Default is 1.
            lambda_factor (float): factor defining the scale of lambda. Default is 0.1.
            params_init (np.ndarray(float)): array of shape (n_cause_groups + n_effect_groups,) containing the
                initial weights of the joint space, cause groups first; it is rescaled to unit norm and cannot
                contain zeros. Default is None, for which the weights are all equal.
            optimizer_name (str): name of the optimizer, 'adam' (default) or 'sgd'.
            learning_rate (float): value of the learning rate. Default is 1e-2.
            learning_rate_decay (str): schedule to damp the learning rate: cosine decay ("cos") or constant
                learning rate (None). Default is None.
            learning_rate_final (float): final value of the learning rate with the "cos" schedule. Default is
                None, for which the learning rate is damped to zero.
            discard_close_ind (int): defines the "close" samples, discarded in all the optimizations; it is a
                number of time steps with 'time_series' and a number of trajectories with 'time_series_ensemble',
                as in the method 'optimize_present_to_future' of the CausalGraph class. Default is 0.
            num_epochs_average (int): number of final training epochs over which the weights are averaged to
                define the final weights. Default is 1.

        Returns:
            imbalance_gains (list(dict)): for each time lag, the dictionary returned by the function
                'return_diff_imbalance_gain'. Also accessible as attribute of the DiffImbalanceGain object.
        """
        cause_variables = self._check_variables(cause_variables, "cause_variables")
        effect_variables = self._check_variables(effect_variables, "effect_variables")
        if np.intersect1d(cause_variables, effect_variables).size > 0:
            raise ValueError(
                "cause_variables and effect_variables must not share variables."
            )
        time_lags = np.atleast_1d(time_lags)
        if time_lag_max is None:
            time_lag_max = np.max(time_lags, initial=0)
        if (
            time_lags.size == 0
            or not np.issubdtype(time_lags.dtype, np.integer)
            or np.min(time_lags) < 1
            or np.max(time_lags) > time_lag_max
        ):
            raise ValueError(
                "time_lags must contain positive integers, not larger than time_lag_max."
            )
        if embedding_dim < 1 or embedding_time < 1:
            raise ValueError("embedding_dim and embedding_time must be positive.")

        t0s, discard_close_samples = self._select_times(
            num_samples,
            [time_lag_max],
            embedding_dim,
            embedding_dim,
            embedding_time,
            discard_close_ind,
        )
        shifts = -embedding_time * np.arange(embedding_dim)
        cause_present = self._extract_samples(t0s, shifts, cause_variables)
        effect_present = self._extract_samples(t0s, shifts, effect_variables)
        # columns ordered by variable, then time shift
        period_cause, period_effect = None, None
        if self.periods is not None:
            period_cause = np.repeat(self.periods[cause_variables], embedding_dim)
            period_effect = np.repeat(self.periods[effect_variables], embedding_dim)

        imbalance_gains = []
        for tau in time_lags:
            imbalance_gains.append(
                return_diff_imbalance_gain(
                    cause_present=cause_present,
                    effect_present=effect_present,
                    effect_future=self._extract_samples(
                        t0s, tau + shifts, effect_variables
                    ),
                    k=k,
                    discard_close_ind=discard_close_samples,
                    num_epochs=num_epochs,
                    params_init=params_init,
                    cause_params_groups=cause_params_groups,
                    effect_params_groups=effect_params_groups,
                    period_cause=period_cause,
                    period_effect=period_effect,
                    batches_per_epoch=batches_per_epoch,
                    track_full_loss=track_full_loss,
                    seed=self.seed,
                    l1_strength=l1_strength,
                    point_adapt_lambda=point_adapt_lambda,
                    lambda_factor=lambda_factor,
                    optimizer_name=optimizer_name,
                    learning_rate=learning_rate,
                    learning_rate_decay=learning_rate_decay,
                    learning_rate_final=learning_rate_final,
                    num_epochs_average=num_epochs_average,
                )
            )
        self.imbalance_gains = imbalance_gains
        return imbalance_gains

    def _check_variables(self, variables, name):
        """Return the indices of the variables as a non-empty array of integers between 0 and D-1."""
        variables = np.atleast_1d(variables)
        if (
            variables.ndim != 1
            or variables.size == 0
            or not np.issubdtype(variables.dtype, np.integer)
            or variables.min() < 0
            or variables.max() >= self.num_variables
        ):
            raise ValueError(
                f"{name} must contain indices of variables between 0 and {self.num_variables - 1}."
            )
        return variables
