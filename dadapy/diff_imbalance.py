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
The *diff_imbalance* module contains the *DiffImbalance* class, implemented with JAX.

The only method supposed to be called by the user is 'train', which carries out the automatic optimization ot the
Differential Information as a function of the weights of the features in the first distance space.
The code can be runned on gpu using the command
    jax.config.update('jax_platform_name', 'gpu') # set 'cpu' or 'gpu'
"""

import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm.auto import tqdm

# OPTIMIZABLE DISTANCE FUNCTIONS
# (here new functions may be added for purposes beyond feature selection)
# ----------------------------------------------------------------------------------------------


# for feature selection
@partial(jax.jit, static_argnames="params_groups")
def _compute_dist2_matrix_scaling(
    params, batch_rows, batch_columns, periods=None, params_groups=None
):
    """Computes the (squared) Euclidean distance matrix between points in 'batch_rows' and points in 'batch_columns'.

    The features of the points are scaled by the weights in 'params', such that the distance between
    point i in batch_rows and point j in batch_columns is computed as
        dist2_matrix[i,j] = ((batch_rows[i,:] - batch_columns[j,:])**2).sum(axis=-1)

    Args:
        params (jnp.array(float)): array of shape (n_params,). If parmas_groups is None, n_params == n_features.
        batch_rows (jnp.array(float)): matrix of shape (n_points_rows, n_features).
        batch_columns (jnp.array(float)): matrix of shape (n_points_columns, n_features).
        periods (jnp.array(float)): array of shape (n_features,) for computing distances between periodic
            features by applying PBCs. If only a subset of features is periodic, the entries of 'periods' for the
            nonperiodic features should be set to zero. Default is None, for which PBCs are not applied.
        params_groups (jnp.array(int)): array of shape (n_params,) containing at position i the number of features
            that share the same weight params[i], using the same order of the columns in batch_rows and batch_columns.
            If params_groups is None, no weight sharing is enforced.
    Returns:
        dist2_matrix (jnp.array(float)): array of shape (n_points_rows, n_features) containing the square Euclidean
            distances between all points in 'batch_rows' and all points in 'batch_columns'.
    """
    diffs = batch_rows[:, jnp.newaxis, :] - batch_columns[jnp.newaxis, :, :]
    if periods is not None:
        periodic_mask = periods > 0  # only shift periodic features
        periodic_shifts = (
            jnp.round(diffs / jnp.where(periodic_mask, periods, 1.0)) * periods
        )
        diffs -= jnp.where(periodic_mask, periodic_shifts, 0.0)

    params_repeated = +params
    if params_groups is not None:
        params_repeated = jnp.repeat(params, np.array(params_groups))

    diffs *= params_repeated[jnp.newaxis, jnp.newaxis, :]
    dist2_matrix = jnp.sum(diffs * diffs, axis=-1)
    return dist2_matrix


# CLASS TO OPTIMIZE THE DIFFERENTIAL INFORMATION IMBALANCE
# ----------------------------------------------------------------------------------------------


class DiffImbalance:
    """Carries out the optimization of the DII(A(w)->B) with respect to the weights in the first distance space.

    The class 'DiffImbalance' supports two schemes for setting the smoothing parameter lambda, which tunes the
    size of neighborhoods in space A. In both schemes lambda can be epoch-dependent, i.e. decreased during the
    training according to a cosine decay between 'init' and 'final' values. The schemes are:

        1. Adaptive: lambda is equal for all the points and is set to a fraction (given by lambda_factor,
        default is 1/10) of the *average* square distance of k-th neighbors.

        Example:
            point_adapt_lambda: False
            k_init: 10
            k_final: 1
            lambda_factor=1/10

        2. Point-adaptive: lambda is different for each point. For point i, it is set to a fraction of the
        square distance between i and its k-th neighbor.

        Example:
            point_adapt_lambda: True
            k_init: 10
            k_final: 1
            lambda_factor=1/10

    As a rule of thumb, we suggest to set k_init and k_final to ~5% of the points in the data set, if mini-
    batches are not employed, or to ~5% of the points within each mini-batch, if they are employed.

    Attributes:
        data_A (np.array(float), jnp.array(float)): feature space A, matrix of shape (n_points, n_features_A).
        data_B (np.array(float), jnp.array(float)): feature space B, matrix of shape (n_points, n_features_B).
        distances_B (np.array(float), jnp.array(float)): distance matrix in space B, of shape (n_points, n_points).
            Default is None, for which distances are computed from the features in data_B.
        periods_A (np.array(float), jnp.array(float)): array of shape (n_features_A,), periods of features A.
            Default is None, which means that features A are treated as nonperiodic. If not all features are
            periodic, the entries of the nonperiodic ones should be set to 0.
        periods_B (np.array(float), jnp.array(float)): array of shape (n_features_B,), periods of features B.
            Default is None, which means that features B are treated as nonperiodic. If not all features are
            periodic, the entries of the nonperiodic ones should be set to 0.
        num_epochs (int): number of training epochs. Default is 200.
        batches_per_epoch (int): number of minibatches; must be a divisor of n_points. Each weight update is
            carried out by computing the DII gradient over n_points / batches_per_epoch points. Default is 1,
            which means that the gradient is computed over all the available points (batch GD).
        seed (int): seed of JAX random generator, default is 0. Different seeds determine different mini-batch
            partitions.
        l1_strength (float): strength of the L1 regularization (LASSO) term. Default is 0.
        gradient_clip_value (float): maximum norm for gradient clipping. If 0, no clipping is
            applied. Default is 0. This is useful when weights are sometimes automatically set to NaN and
            there can be gradient explosions.
        point_adapt_lambda (bool): whether to use a global smoothing parameter lambda for the c_ij coefficients
            in the DII (if False), or a different parameter for each point (if True). Default is True.
        k_init (int): initial rank of neighbors used to set lambda. Ranks are defined starting from 1. If
            batches_per_epoch > 1, neighbors are recomputed within each mini-batch. Default is 1.
        k_final (int): final rank of neighbors used to set lambda. If batches_per_epoch > 1, neighbors are
            recomputed within each mini-batch. Default is 1.
        lambda_factor (float): factor defining the scale of lambda. Default is 0.1.
        params_init (np.array(float), jnp.array(float)): array of shape (n_params,) containing the initial
            values of the scaling weights to be optimized. If params_groups is set to None, each feature is
            scaled by an independent optimization parameter, so n_params == n_features_A. If params_init is None,
            the initial scaling parameters are set to [0.1, 0.1, ..., 0.1].
        params_groups (np.array(int), jnp.array(int)): array of shape (n_params,) containing at position i the
            number of features that share the same weight in params_init[i], using the same order of the columns
            in data_A. If params_groups = [3, 2, 4], for example, the first 3 features in space A will share a
            common weight, the following 2 features will share a second common weight, and the last 4 features
            will also be scaled by a common optimization parameter. params_groups should satisfy the constraint
            sum(params_groups) == n_features_A. If params_groups is None, no weight sharing is enforced.
        optimizer_name (str): name of the optimizer, calling the Optax library. Possible choices are 'sgd'
            (default), 'adam' and 'adamw'. See https://optax.readthedocs.io/en/latest/api/optimizers.html for
            additional details.
        learning_rate (float): value of the learning rate. Default is 1e-2.
        learning_rate_decay (str): schedule to damp the learning rate to zero starting from the value provided
            with the attribute learning_rate. The available schedules are: cosine decay ("cos"), exponential
            decay ("exp"; the initial learning rate is halved every 10 steps), or constant learning rate (None).
            Default is None (constant learning rate).
        num_points_rows (int): number of points sampled from the rows of rank and distance matrices. In case of large
            datasets, choosing num_points_rows < n_points can significantly speed up the training. The default is
            None, for which num_points_rows == n_points.
    """

    def __init__(
        self,
        data_A,
        data_B,
        distances_B=None,
        periods_A=None,
        periods_B=None,
        num_epochs=200,
        batches_per_epoch=1,
        seed=0,
        l1_strength=0.0,
        point_adapt_lambda=True,
        k_init=1,
        k_final=1,
        lambda_factor=0.1,
        params_init=None,
        params_groups=None,
        optimizer_name="sgd",
        learning_rate=1e-2,
        learning_rate_decay=None,
        num_points_rows=None,
        gradient_clip_value=0.0,
    ):
        """Initialise the DiffImbalance class."""
        self.nfeatures_A = data_A.shape[1]
        if distances_B is None:  # space B provided as features
            self.nfeatures_B = data_B.shape[1]
            assert data_A.shape[0] == data_B.shape[0], (
                f"Space A has {data_A.shape[0]} samples "
                + f"while space B has {data_B.shape[0]} samples."
            )
        else:  # space B provided as distances
            if data_B is not None:
                warnings.warn(
                    f"Argument distances_B is not None; data_B will be ignored."
                )
            # self.distances_B = jnp.array(distances_B)
            assert (
                distances_B.shape[0] == distances_B.shape[1]
            ), f"Argument distances_B should be a square matrix, while it has shape {distances_B.shape}"
            assert data_A.shape[0] == distances_B.shape[0], (
                f"Number of points in data_A ({data_A.shape[0]}) and distances_B ({distances_B.shape[0]})"
                + f" do not match."
            )
        self.nparams = self.nfeatures_A if params_groups is None else len(params_groups)

        # initialize jax random generator
        self.key = jax.random.PRNGKey(seed)
        self.key, subkey = jax.random.split(self.key, num=2)

        # initialize spaces A and B
        self.data_A = data_A
        self.data_B = data_B
        self.distances_B = distances_B

        # option to speed up DII calculation by decimating rows (rectangular distance matrices)
        if num_points_rows is not None:
            assert num_points_rows < self.data_A.shape[0], (
                f"num_points_rows ({num_points_rows}) should be smaller than the number "
                + f"of points in the data set ({self.data_A.shape[0]}) or set to None."
            )
            # decimate rows but not columns, and keep same indices in upper left square matrix
            indices_rows = jax.random.choice(
                subkey,
                jnp.arange(data_A.shape[0]),
                shape=(num_points_rows,),
                replace=False,
            )
            indices_columns = jnp.delete(jnp.arange(data_A.shape[0]), indices_rows)
            indices_columns = jnp.concatenate((indices_rows, indices_columns))
        else:
            indices_rows = jnp.arange(data_A.shape[0])
            indices_columns = +indices_rows

        self.data_A_rows = data_A[indices_rows]
        self.data_A_columns = data_A[indices_columns]
        if self.distances_B is None:  # space B provided as features
            self.data_B_rows = data_B[indices_rows]
            self.data_B_columns = data_B[indices_columns]
        else:  # space B provided as distances
            self.distances_B_subsampled = self.distances_B[indices_rows][
                :, indices_columns
            ]

        self.nrows = self.data_A_rows.shape[0]
        self.ncolumns = self.data_A_columns.shape[0]
        self.periods_A = (
            jnp.ones(self.nfeatures_A) * jnp.array(periods_A)
            if periods_A is not None
            else periods_A
        )
        if self.distances_B is None:  # space B provided as features
            self.periods_B = (
                jnp.ones(self.nfeatures_B) * jnp.array(periods_B)
                if periods_B is not None
                else periods_B
            )
        self.num_epochs = num_epochs
        self.batches_per_epoch = batches_per_epoch
        self.l1_strength = l1_strength
        self.gradient_clip_value = gradient_clip_value
        self.point_adapt_lambda = point_adapt_lambda
        self.k_init = k_init
        self.k_final = k_final
        self.lambda_factor = lambda_factor
        if params_init is not None:
            self.params_init = jnp.array(params_init, dtype=float)
        else:
            self.params_init = 0.1 * jnp.ones(self.nparams)
        self.params_groups = params_groups
        if params_groups is not None:
            self.params_groups = tuple(params_groups)
        self.params_final = None
        self.params_training = None
        self.imb_final = None
        self.imbs_training = None
        self.error_final = None
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.num_points_rows = num_points_rows
        self.mask = None

        self.state = None
        self._distance_A = _compute_dist2_matrix_scaling  # TODO: assign other functions if other distances d_A are chosen

        # generic checks and warnings
        assert self.nrows >= batches_per_epoch, (
            f"Cannot extract {batches_per_epoch} minibatches "
            + f"from {self.nrows} samples."
        )
        assert (
            self.k_init is not None and self.k_final is not None
        ), f"Provide values of 'k_init' and 'k_final' to compute lambda adaptively."
        if self.k_init > 100:
            warnings.warn(
                f"For efficiency reasons the maximum value for 'k_init' is 100, while you set it to {self.k_init}.\n"
                + f"The run will continue with 'k_init = 100'"
            )
            self.k_init = 100
        assert (
            self.k_init >= self.k_final and self.k_final > 0
        ), f"'k_init' and 'k_final' must satisfy: k_init >= k_final >= 1."
        assert isinstance(k_init, int) and isinstance(
            k_final, int
        ), f"'k_init' and 'k_final' must be positive integers."
        if self.params_groups is not None:
            n_vars = np.sum(self.params_groups)
            assert n_vars == self.nfeatures_A, (
                f"Number of elements in 'params_groups' ({n_vars}) does not match the number "
                + f"of features in space A ({self.nfeatures_A})."
            )
        assert self.params_init.shape[0] == self.nparams, (
            f"With your inputs ('data_A' and 'params_groups'), 'params_init' should contain {self.nparams} weights, "
            + f"while it contains {self.params_init.shape[0]} weights."
        )

        # create jitted functions
        self._create_functions()

        # pre-compute ranks B to speed up training
        if self.distances_B is None:  # input B provided as features
            self.ranks_B = self._compute_rank_matrix(
                batch_rows=self.data_B_rows,
                batch_columns=self.data_B_columns,
                periods=self.periods_B,
            )
        else:  # input B provided as features
            self.ranks_B = self.distances_B_subsampled.argsort(axis=1).argsort(axis=1)

        # set method to compute lambda (adaptive or point-adaptive)
        if point_adapt_lambda:
            self.lambda_method = self._compute_point_adapt_lambdas
        else:
            self.lambda_method = self._compute_adapt_lambda

    def _create_functions(self):
        def _compute_rank_matrix(batch_rows, batch_columns, periods):
            """Computes the matrix of ranks for the target space B.

            Args:
                batch_rows (jnp.array(float)): matrix of shape (n_points_rows, n_features_B), containing
                    points labelling the rank matrix rows.
                batch_columns (jnp.array(float)): matrix of shape (n_points_columns, n_features_B), containing
                    points labelling the rank matrix columns.
                periods (jnp.array(float)): array of shape (n_features_B,), containing the periods of features
                    in space B. PBCs are not applied for feature i if periods[i] == 0, or if periods == None.

            Returns:
                rank_matrix (jnp.array(float)): matrix of shape (n_points_rows, n_points_columns), defining the
                    target distance ranks in space B. Ranks start from 1, and are 0 only for a point with respect
                    to itself (when a point appears both in batch_rows and batch_columns).
            """
            diffs = batch_rows[:, jnp.newaxis, :] - batch_columns[jnp.newaxis, :, :]
            if periods is not None:
                periodic_mask = periods > 0  # only shift periodic features
                periodic_shifts = (
                    jnp.round(diffs / jnp.where(periodic_mask, periods, 1.0)) * periods
                )
                diffs -= jnp.where(periodic_mask, periodic_shifts, 0.0)
            dist2_matrix = jnp.sum(diffs * diffs, axis=-1)
            rank_matrix = dist2_matrix.argsort(axis=1).argsort(axis=1)
            return rank_matrix

        def _cosine_decay_func(start_value, final_value, step):
            """Implements a cosine decay during the training.

            The arguments start_value and final_value can be values of the learning rate or of the
            neighbor order used to compute lambda in the adaptive and point-adaptive schemes.

            Args:
                start_value (float): initial value.
                final_value (float): final value.
                step (int): number of current gradient descent step.

            Returns:
                cosine (float): value of the cosine interpolating start_value at step 0 and final_value
                    at the last training step.
            """
            x = jnp.pi / (self.num_epochs * self.batches_per_epoch) * step
            cosine = (start_value - final_value) * (jnp.cos(x) + 1) / 2 + final_value
            return cosine

        def _compute_point_adapt_lambdas(dist2_matrix, step=None, k=None):
            """Computes lambda parameters with the point-adaptive scheme, according to the current value of k.

            Args:
                dist2_matrix (jnp.array(float)): matrix of shape (n_points_rows, n_points_columns), containing
                    the squared distances in space A at the current training step.
                step (int): number of current gradient descent step, from which the current value of 'k' to
                    compute lambda adaptively is obtained.
                k (int): neighbor order to set lambda adaptively (alternative to step).

            Returns:
                current_lambdas (jnp.array(float)): array of shape (n_points_rows,), containing a value of lambda
                    computed adaptively for each point, as the fraction ('lambda_factor', default: 1/10) of the
                    squared distance of the neighbor of order k.
            """
            if step is not None:
                current_k = jnp.rint(
                    _cosine_decay_func(
                        start_value=self.k_init, final_value=self.k_final, step=step
                    )
                ).astype(int)
            elif k is not None:
                current_k = k
            # take the k_max_allowed smallest distances with negative sign
            k_max_allowed = 100
            if dist2_matrix.shape[1] < k_max_allowed:
                k_max_allowed = dist2_matrix.shape[1]
            smallest_dist2, _ = jax.lax.top_k(-dist2_matrix, k_max_allowed)
            current_lambdas = -smallest_dist2[:, current_k - 1] * self.lambda_factor

            # DON'T DELETE: Adaptive scheme of cython code
            # diffs_dists_2nd_1st = -smallest_dist2[:, 1] + smallest_dist2[:, 0]
            # current_lambdas = 0.5*(diffs_dists_2nd_1st.min() + diffs_dists_2nd_1st.mean())

            return current_lambdas

        def _compute_adapt_lambda(dist2_matrix, step=None, k=None):
            """Computes smoothing parameter lambda with adaptive scheme.

            Args:
                dist2_matrix (jnp.array(float)): matrix of shape (n_points_rows, n_points_columns), containing
                    the squared distances in space A at the current training step.
                step (int): number of current gradient descent step, from which the current value of 'k' to
                    compute lambda adaptively is obtained.
                k (int): neighbor order to set lambda adaptively (alternative to step).

            Returns:
                current_lambda (jnp.array(float)): array of shape (n_points_rows,), containing the same value of lambda
                    for all points, computed as the fraction (1/10) of the *average* squared distance of the neighbor
                    of order k.
            """
            current_lambda = _compute_point_adapt_lambdas(
                dist2_matrix, step, k
            ).mean() * jnp.ones(dist2_matrix.shape[0])
            return current_lambda

        def _compute_training_diff_imbalance(
            params, batch_A_rows, batch_A_columns, batch_B_ranks, step
        ):
            """Computes the Differentiable Information Imbalance (DII) at the current step of the training.

            Args:
                params (jnp.array(float)): array of shape (n_features_A,) of the current feature weights.
                batch_A_rows (jnp.array(float)): matrix of shape (n_points_rows, n_features_A), containing
                    points labelling the distance matrix rows.
                batch_A_columns (jnp.array(float)): matrix of shape (n_points_columns, n_features_A), containing
                    points labelling the distance matrix columns.
                batch_B_ranks (jnp.array(float)): matrix of shape (n_points_rows, n_points_columns), containing
                    the pre-computed target ranks in space B.
                step (int): number of current gradient descent step.

            Returns:
                diff_imbalance (float): current value of the DII.
            """
            dist2_matrix_A = self._distance_A(  # compute distance matrix A
                params=params,
                batch_rows=batch_A_rows,
                batch_columns=batch_A_columns,
                periods=self.periods_A,
                params_groups=self.params_groups,
            )
            N = dist2_matrix_A.shape[0]
            max_rank = dist2_matrix_A.shape[1] - 1

            # set distance of a point with itself to large number
            dist2_matrix_A = dist2_matrix_A.at[jnp.arange(N), jnp.arange(N)].set(
                jnp.max(dist2_matrix_A) + 1e6
            )

            lambdas = self.lambda_method(
                dist2_matrix=dist2_matrix_A, step=step
            )  # compute lambda values
            c_matrix = (
                jax.nn.softmax(  # N.B. diagonal elements already numerically zero
                    -dist2_matrix_A
                    / lambdas[
                        :, jnp.newaxis
                    ],  # jax.lax.stop_gradient(lambdas[:, jnp.newaxis])
                    axis=1,
                )
            )

            # DON'T DELETE: Alternative definition of c_ij coefficients (sigmoid instead of softmax)
            # c_matrix = jax.nn.sigmoid(
            #    (lambdas[:, jnp.newaxis] - dist2_matrix_A)/(self.lambda_factor * lambdas[:, jnp.newaxis])
            # )

            # compute DII
            conditional_ranks = jnp.sum(batch_B_ranks * c_matrix, axis=1)
            diff_imbalance = 2.0 / (max_rank + 1) * jnp.sum(conditional_ranks) / N

            # DON'T DELETE: analytical gradient of the DII (without differentiating lambda)
            # diffs_squared = ((batch_A_rows[:,jnp.newaxis,:] - batch_A_columns[jnp.newaxis,:,:])
            #                *(batch_A_rows[:,jnp.newaxis,:] - batch_A_columns[jnp.newaxis,:,:])) # shape (nrows, ncols, D)
            # second_term = (c_matrix[:,:,jnp.newaxis] * diffs_squared).sum(axis=1, keepdims=True)
            # grad_imbalance = (
            #    4.0 * params / (N * (self.max_rank + 1))
            #    * jnp.sum((batch_B_ranks * c_matrix)[:,:,jnp.newaxis] / lambdas[:,jnp.newaxis,jnp.newaxis]
            #    * (-diffs_squared + second_term), axis=(0,1))
            # )
            return diff_imbalance

        def _compute_final_diff_imbalance_and_error(
            params, batch_A_rows, batch_A_columns, batch_B_ranks, k
        ):
            """Computes the Differentiable Information Imbalance (DII) and its error.

            Args:
                params (jnp.array(float)): array of shape (n_features_A,) of the current feature weights.
                batch_A_rows (jnp.array(float)): matrix of shape (n_points_rows, n_features_A), containing
                    points labelling the distance matrix rows.
                batch_A_columns (jnp.array(float)): matrix of shape (n_points_columns, n_features_A), containing
                    points labelling the distance matrix columns.
                batch_B_ranks (jnp.array(float)): matrix of shape (n_points_rows, n_points_columns), containing
                    the pre-computed target ranks in space B.
                k (int): neighbor order to set lambda adaptively.

            Returns:
                diff_imbalance (float): value of the DII.
                error_imbalance (float): error associated to the DII.
            """
            dist2_matrix_A = self._distance_A(  # compute distance matrix A
                params=params,
                batch_rows=batch_A_rows,
                batch_columns=batch_A_columns,
                periods=self.periods_A,
                params_groups=self.params_groups,
            )
            N = dist2_matrix_A.shape[0]
            max_rank = dist2_matrix_A.shape[1]
            lambdas = self.lambda_method(
                dist2_matrix=dist2_matrix_A, k=k
            )  # compute lambda values
            c_matrix = jax.nn.softmax(
                -dist2_matrix_A
                / lambdas[
                    :, jnp.newaxis
                ],  # jax.lax.stop_gradient(lambdas[:,jnp.newaxis])
                axis=1,
            )

            # DON'T DELETE: compute standard Information Imbalance
            # batch_A_ranks = dist2_matrix_A.argsort(axis=1).argsort(axis=1) + 1
            # mask_A = (batch_A_ranks <= k)
            # conditional_ranks = jnp.where(mask_A, 1.0, 0.0) * batch_B_ranks
            # conditional_ranks = conditional_ranks.sum(axis=-1) / k
            # values_average = 2.0 / (max_rank + 1) * conditional_ranks

            # compute DII and error
            values_average = (
                2.0 / (max_rank + 1) * jnp.sum(batch_B_ranks * c_matrix, axis=1)
            )
            diff_imbalance = jnp.mean(values_average)
            error_imbalance = jnp.std(values_average, ddof=1) / jnp.sqrt(N)

            return diff_imbalance, error_imbalance

        def _compute_final_diff_imbalance(
            params, batch_A_rows, batch_A_columns, batch_B_ranks, k
        ):
            """Computes the Differentiable Information Imbalance (DII) without providing the error.

            Args:
                params (jnp.array(float)): array of shape (n_features_A,) of the current feature weights.
                batch_A_rows (jnp.array(float)): matrix of shape (n_points_rows, n_features_A), containing
                    the points labelling the distance matrix rows.
                batch_A_columns (jnp.array(float)): matrix of shape (n_points_columns, n_features_A), containing
                    the points labelling the distance matrix columns.
                batch_B_ranks (jnp.array(float)): matrix of shape (n_points_rows, n_points_columns), containing
                    the pre-computed target ranks in space B.
                k (int): neighbor order to set lambda adaptively.

            Returns:
                diff_imbalance (float): value of the DII.
            """
            dist2_matrix_A = self._distance_A(  # compute distance matrix A
                params=params,
                batch_rows=batch_A_rows,
                batch_columns=batch_A_columns,
                periods=self.periods_A,
                params_groups=self.params_groups,
            )
            N = dist2_matrix_A.shape[0]
            max_rank = dist2_matrix_A.shape[1] - 1

            # set distance of a point with itself to large number
            dist2_matrix_A = dist2_matrix_A.at[jnp.arange(N), jnp.arange(N)].set(
                jnp.max(dist2_matrix_A) + 1e6
            )
            # apply mask to column indices around the row index
            if self.mask is not None:
                dist2_matrix_A = dist2_matrix_A[self.mask].reshape(
                    (dist2_matrix_A.shape[0], -1)
                )
                max_rank = dist2_matrix_A.shape[1]
            lambdas = self.lambda_method(
                dist2_matrix=dist2_matrix_A, k=k
            )  # compute lambda values
            c_matrix = jax.nn.softmax(  # N.B. diagonal elements already numerically zero if mask is None
                -dist2_matrix_A / lambdas[:, jnp.newaxis],
                axis=1,
            )

            # compute DII
            conditional_ranks = jnp.sum(batch_B_ranks * c_matrix, axis=1)
            diff_imbalance = 2.0 / (max_rank + 1) * jnp.sum(conditional_ranks) / N
            return diff_imbalance

        def _train_step(state, batch_A_rows, batch_A_columns, batch_B_ranks):
            """Performs a single gradient descent step in the optimization of the DII.

            Args:
                state (flax.training.train_state.TrainState object): current training state.
                batch_A_rows (jnp.array(float)): matrix of shape (n_points_rows, n_features_A), containing
                    the points labelling the distance matrix rows.
                batch_A_columns (jnp.array(float)): matrix of shape (n_points_columns, n_features_A), containing
                    the points labelling the distance matrix columns.
                batch_B_ranks (jnp.array(float)): matrix of shape (n_points_rows, n_points_columns), containing
                    the pre-computed target ranks in space B.

            Returns:
                state_new (flax.training.train_state.TrainState object): new training state after optimizer step
                imb (flat): new value of the DII after optimizer step.
            """
            loss_fn = lambda params: _compute_training_diff_imbalance(
                params=params,
                batch_A_rows=batch_A_rows,
                batch_A_columns=batch_A_columns,
                batch_B_ranks=batch_B_ranks,
                step=state.step,
            )
            # Get loss and gradient
            imb, grads = jax.value_and_grad(loss_fn)(state.params)

            # Update parameters
            state = state.apply_gradients(grads=grads)
            norm_init = jnp.sqrt((self.params_init**2).sum())
            norm_now = jnp.sqrt((state.params**2).sum())
            # Scale weight vector to original norm
            state = state.replace(params=norm_init / norm_now * state.params)

            # Apply L1 penalty
            if self.l1_strength != 0:
                current_lr = self.lr_schedule(state.step)

                # (GD clipping, B. Carpenter et al, 2008)
                state = state.replace(
                    params=jnp.where(state.params > 0, 1.0, 0.0)
                    * jnp.maximum(0, state.params - current_lr * self.l1_strength)
                    + jnp.where(state.params < 0, 1.0, 0.0)
                    * jnp.minimum(0, state.params + current_lr * self.l1_strength)
                )

                # DON'T DELETE: Soft version of GD clipping
                # candidate_params = (
                #    state.params
                #    - jnp.sign(state.params) * current_lr * self.l1_strength
                # )
                # state = state.replace(
                #    params=state.params
                #    * (1.0 - jnp.where(state.params * candidate_params < 0, 1.0, 0.0))
                # )

                # Scale weight vector to original norm
                norm_now = jnp.sqrt((state.params**2).sum())
                state = state.replace(params=norm_init / norm_now * state.params)
            return state, imb

        # jit compilation of functions
        self._compute_rank_matrix = jax.jit(_compute_rank_matrix)
        self._cosine_decay_func = jax.jit(_cosine_decay_func)
        self._compute_point_adapt_lambdas = jax.jit(_compute_point_adapt_lambdas)
        self._compute_adapt_lambda = jax.jit(_compute_adapt_lambda)
        self._compute_training_diff_imbalance = jax.jit(
            _compute_training_diff_imbalance
        )
        self._compute_final_diff_imbalance_and_error = jax.jit(
            _compute_final_diff_imbalance_and_error
        )
        self._compute_final_diff_imbalance = jax.jit(_compute_final_diff_imbalance)
        self._train_step = jax.jit(_train_step)

    def _return_mask(self, npoints, discard_close_ind):
        """Returns a square boolean mask with False on the diagonals, and True elsewhere.

        Args:
            npoints (int): number of rows and columns of the mask matrix.
            discard_close_ind (int): defines the diagonals filled with False, with offset between
                -discard_close_ind (below the main diagonal) and +discard_close_ind (above the main
                diagonal).

        Returns:
            mask (jnp.array(float)): square boolean matrix of shape (npoints, npoints).
        """
        mask = jnp.abs(
            jnp.arange(npoints)[:, jnp.newaxis] - jnp.arange(npoints)[jnp.newaxis, :]
        )
        mask = (mask > discard_close_ind).astype(jnp.bool)
        # more columns than necessary discarded for starting and final rows, for shape compatibility
        first_rows = jnp.concatenate(
            (
                jnp.zeros(2 * discard_close_ind + 1),
                jnp.ones(npoints - 2 * discard_close_ind - 1),
            ),
            dtype=bool,
        )
        last_rows = jnp.concatenate(
            (
                jnp.ones(npoints - 2 * discard_close_ind - 1),
                jnp.zeros(2 * discard_close_ind + 1),
            ),
            dtype=bool,
        )
        mask = mask.at[:discard_close_ind].set(first_rows)
        mask = mask.at[-discard_close_ind:].set(last_rows)
        return mask

    def _return_nn_indices(self, discard_close_ind=0):
        """
        Returns indices of the nearest neighbor of each point.

        Args:
            discard_close_ind (int): given any point i, defines the "close" points (following the labelling order
                along axis=0 of data_A and data_B) that are known to be significantly correlated with i. For example,
                this may occur when the data set is a time series, and axis=0 is the time dimension. For each point i,
                distances between i and points within the time window [i-discard_close_ind, i+discard_close_ind] are
                discarded. Default is 0, for which no distances between "time-correlated" points are discarded.

        Returns:
            nn_indices (np.array(float)): array of the nearest neighbors indices: nn_indices[i] is the index of the
                column with value 1 in the rank matrix.
        """
        rank_matrix = self._compute_rank_matrix(
            batch_rows=self.data_A_rows,
            batch_columns=self.data_A_columns,
            periods=self.periods_A,
        )
        npoints = rank_matrix.shape[0]
        # discard diagonal elements
        rank_matrix = rank_matrix.at[jnp.arange(npoints), jnp.arange(npoints)].set(
            npoints + 1
        )

        # construct and apply mask to discard distances between "close" points
        if discard_close_ind > 0:
            mask = self._return_mask(
                npoints=rank_matrix.shape[0], discard_close_ind=discard_close_ind
            )
            rank_matrix = rank_matrix[mask].reshape((rank_matrix.shape[0], -1))
            rank_matrix = rank_matrix.argsort(axis=1).argsort(axis=1) + 1

        nn_indices = jnp.argmin(rank_matrix, axis=1)
        return nn_indices

    def _train_epoch(self, key):
        """Performs the training for a single epoch.

        Args:
            key (jax.random.PRNGKey): key for the JAX pseudo-random number generator (PRNG).

        Returns:
            params (jnp.array(float)): array of shape (n_features_A,) containing the
                weights at the last step of the current training epoch. The single mini-batch
                updates are not returned.
            imb (float): value of the DII at the last step of the current training epoch.
        """
        # ----------------------------MINI-BATCH GD----------------------------
        if self.batches_per_epoch > 1:
            all_batch_indices = jnp.split(
                jax.random.permutation(key, self.nrows), self.batches_per_epoch
            )

            # mini-batch GD (subsample both rows and columns)
            for batch_indices in all_batch_indices:
                self.state, imb = self._train_step(
                    self.state,
                    self.data_A_rows[batch_indices],
                    self.data_A_columns[batch_indices],
                    self.ranks_B[batch_indices][:, batch_indices]
                    .argsort(axis=1)
                    .argsort(axis=1),
                )
            # DON'T DELETE: Alternative method for mini-batch GD (only subsample rows)
            # for i_batch, batch_indices in enumerate(all_batch_indices):
            #    ordered_column_indices = np.ravel(
            #        np.delete(all_batch_indices, i_batch, axis=0)
            #    )
            #    ordered_column_indices = np.append(
            #        batch_indices, ordered_column_indices
            #    )
            #    self.state, imb = self._train_step(
            #        self.state,
            #        self.data_A_rows[batch_indices],
            #        self.data_A_columns[ordered_column_indices],
            #        self.ranks_B[batch_indices][:, ordered_column_indices],
            #    )

        # -----------------------------BATCH GD----------------------------
        else:
            self.state, imb = self._train_step(
                self.state,
                self.data_A_rows,
                self.data_A_columns,
                self.ranks_B,
            )
        assert not jnp.isnan(self.state.params).any(), (
            "All weights were set to zero during the optimization. "
            + "Reduce the value of l1_strength."
        )

        return self.state.params, imb

    def _init_optimizer(self):
        """Initializes the optimizer and the training state using the Optax library.

        The function uses the attribute optimizer_name of the DiffImbalance object,
        which can be set to one of the following options: "sgd", "adam", "adamw". For more
        information on these optimizers, see https://optax.readthedocs.io/en/latest/api/optimizers.html.
        """
        if self.optimizer_name.lower() == "adam":
            opt_class = optax.adam
        elif self.optimizer_name.lower() == "adamw":
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == "sgd":
            opt_class = optax.sgd
        else:
            raise ValueError(
                f'Unknown optimizer "{self.optimizer_name.lower()}". Choose among "sgd", "adam" and "adamw".'
            )

        # set the learning rate schedule (cosine decay, exp decay or constant)
        if self.learning_rate_decay == "cos":
            self.lr_schedule = optax.cosine_decay_schedule(
                init_value=self.learning_rate,
                decay_steps=self.num_epochs * self.batches_per_epoch,
            )
        elif self.learning_rate_decay == "exp":
            self.lr_schedule = optax.exponential_decay(
                init_value=self.learning_rate,
                transition_steps=10,
                decay_rate=0.5,
            )
        elif self.learning_rate_decay is None:
            self.lr_schedule = optax.constant_schedule(value=self.learning_rate)
        else:
            raise ValueError(
                f'Unknown learning rate decay schedule "{self.learning_rate_decay}". Choose among None, "cos" and "exp".'
            )
        # Set up optimizer with optional gradient clipping
        if self.gradient_clip_value > 0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.gradient_clip_value),
                opt_class(self.lr_schedule),
            )
        else:
            optimizer = opt_class(self.lr_schedule)

        # Initialize training state
        self.state = train_state.TrainState.create(
            apply_fn=self._distance_A,
            params=self.params_init if self.state is None else self.state.params,
            tx=optimizer,
        )

    def train(self, bar_label=None):
        """Performs the full training of the DII, using the input attributes of the DiffImbalance object.

        Notice that when mini-batches are employed, for efficiency reasons the DII is *not* recomputed
        over the full data set at each training epoch. To access the value of the DII over the full data
        set, use after training the method 'return_final_dii'.

        Args:
            bar_label (str): label on the tqdm training bar, useful when several trains are performed.

        Returns:
            params_training (np.array(float)): matrix of shape (num_epochs+1, n_features_A) containing the
                feature weights during the training, starting from their initialization. Also accessible as
                attribute of the CausalGraph object.
            imbs_training (np.array(float)): array of shape (num_epochs+1,) containing the DII during the
                training. Element imbs_training[i] is the DII computed over the last mini-batch used
                in training epoch i. The same output is accessible as attribute of the CausalGraph object.
        """
        # Initialize optimizer
        self._init_optimizer()

        # Construct output arrays and initialize them using inital weights
        params_training = jnp.empty(shape=(self.num_epochs + 1, self.nparams))
        imbs_training = jnp.empty(shape=(self.num_epochs + 1,))
        batch_indices = jnp.arange(self.nrows // self.batches_per_epoch)

        imb_start = self._compute_training_diff_imbalance(
            params=self.params_init,
            batch_A_rows=self.data_A_rows[batch_indices],
            batch_A_columns=self.data_A_columns[batch_indices],
            batch_B_ranks=self.ranks_B[batch_indices][:, batch_indices]
            .argsort(axis=1)
            .argsort(axis=1),
            step=0,
        )
        # DON'T DELETE: Alternative method for mini-batching (only sample rows)
        # imb_start = self._compute_training_diff_imbalance(
        #    params=self.params_init,
        #    batch_A_rows=self.data_A_rows[batch_indices],
        #    batch_A_columns=self.data_A_columns,
        #    batch_B_ranks=self.ranks_B[batch_indices],
        #    step=0,
        # )
        params_training = params_training.at[0].set(jnp.abs(self.params_init))
        imbs_training = imbs_training.at[0].set(imb_start)

        # Train over different epochs
        desc = "Training"
        if bar_label is not None:
            desc += f" ({bar_label})"
        for epoch_idx in tqdm(range(1, self.num_epochs + 1), desc=desc):
            self.key, subkey = jax.random.split(self.key, num=2)
            params_now, imb_now = self._train_epoch(subkey)
            params_training = params_training.at[epoch_idx].set(jnp.abs(params_now))
            imbs_training = imbs_training.at[epoch_idx].set(imb_now)
        self.params_final = params_training[-1]
        self.params_training = params_training
        self.imbs_training = imbs_training

        return np.array(params_training), np.array(imbs_training)

    def return_final_dii(
        self, compute_error=True, ratio_rows_columns=1, seed=0, discard_close_ind=0
    ):
        """Returns final DII computed over the full data set using the optimal weights.

        If the training was carried out with mini-batches of small size, this method allows computing a better
        estimate of the DII than the final DII value produced by 'train'.
        When 'compute_error=False' and 'discard_close_ind=0', the final DII produced by 'train' is the same computed
        by 'return_final_dii' if the training was performed without mini-batches (batches_per_epoch=1) and without
        row subsampling ('num_points_rows=None').
        The value of k for computing the smoothing parameter lambda is set in order to keep the same ratio k/N used
        in the training phase (if batches_per_epoch > 1, N is the size of mini-batches used during the training).

        Args:
            compute_error (bool): whether to compute the final DII and its error by sampling different points along
                rows and columns of the distance matrix. If False, the final DII is computed using the same points
                along rows and columns, which does not allow for an error estimation. Default is True.
            ratio_rows_columns (float): only read when compute_error is True; defines the ratio between the number
                of points along rows (nrows) and along columns (ncolumns) of distance and rank matrices, in two groups
                randomly sampled.  In general, nrows and ncolumns are determined by solving the equations
                    nrows / ncolumns = ratio_rows_columns,
                    nrows + ncolumns = n_total_points.
                Default is 1, which means that both groups have n_points / 2 elements.
            discard_close_ind (int): given any point i, defines the "close" points (following the labelling order
                along axis=0 of data_A and data_B) that are known to be significantly correlated with i. For example,
                this may occur when the data set is a time series, and axis=0 is the time dimension. If compute_error
                is True, "time-correlated" points are excluded by subsampling the data along axis=0 with stride
                discard_close_ind + 1. If compute_error is False, distances between each point i and points within the
                time window [i-discard_close_ind, i+discard_close_ind] are discarded. Default is 0, for which no
                distances between points close in time are discarded.
            seed (int): seed of JAX random generator, default is 0.

        Returns:
            imb_final (float): final DII, also accessible as attribute of the CausalGraph object.
            error_final (float): error associated to final DII, also accessible as attribute of the CausalGraph object.
                If compute_error is False, error_final is set to None.
        """
        assert self.params_final is not None, "First call the train() method!"
        if compute_error is True and ratio_rows_columns is None:
            raise ValueError(
                "Option 'compute_error==True' requires a value for the argument 'ratio_rows_columns'."
            )
        elif compute_error is False and ratio_rows_columns is not None:
            warnings.warn(
                f"You set 'compute_error' to False; argument 'ratio_rows_columns' will be ignored.\n"
                + f"To suppress this warning set it to None."
            )

        # case 1: compute final DII and its error, using different points for rows and columns
        if compute_error == True:
            # subsample data to remove neighbor correlations, with stride discard_close_ind+1
            data_A = self.data_A
            data_B = self.data_B
            distances_B = self.distances_B
            if discard_close_ind != 0:
                subsamples = jnp.arange(
                    0, self.data_A.shape[0], discard_close_ind + 1, dtype=int
                )
                data_A = data_A[subsamples]
                if self.distances_B is None:
                    data_B = data_B[subsamples]
                else:
                    distances_B = distances_B[subsamples][:, subsamples]

            # Split points in two groups, labelling rows and columns. The number of rows 'nrows'
            # comes from equations nrows / ncols = ratio_rows_columns and nrows + ncols = npoints.
            nrows = int(ratio_rows_columns / (ratio_rows_columns + 1) * data_A.shape[0])
            self.key = jax.random.PRNGKey(seed)  # initialize jax random generator
            self.key, subkey = jax.random.split(self.key, num=2)
            indices_rows = jax.random.choice(
                subkey, jnp.arange(data_A.shape[0]), shape=(nrows,), replace=False
            )
            indices_columns = jnp.delete(jnp.arange(data_A.shape[0]), indices_rows)

            # compute final DII and its error
            if self.distances_B is None:  # space B provided as features
                ranks_B = (
                    self._compute_rank_matrix(
                        batch_rows=data_B[indices_rows],
                        batch_columns=data_B[indices_columns],
                        periods=self.periods_B,
                    )
                    + 1
                )
            else:  # space B provided as distances
                ranks_B = (
                    (distances_B[indices_rows][:, indices_columns])
                    .argsort(axis=1)
                    .argsort(axis=1)
                ) + 1

            # set k to keep same ration k/N used during DII training
            k = int(
                jnp.ceil(
                    self.k_final * self.batches_per_epoch / (discard_close_ind + 1)
                )
            )
            imb_final, error_final = self._compute_final_diff_imbalance_and_error(
                params=self.params_final,
                batch_A_rows=data_A[indices_rows],
                batch_A_columns=data_A[indices_columns],
                batch_B_ranks=ranks_B,
                k=k,
            )

        # case 2: compute final DII only (square distance matrices)
        elif compute_error == False:
            # construct mask to discard distances d[i, i-discard_close_ind:i+discard_close_ind+1], for each i
            mask = None
            self.mask = None
            npoints = self.data_A.shape[0]
            if discard_close_ind != 0:
                mask = self._return_mask(
                    npoints=npoints, discard_close_ind=discard_close_ind
                )
            self.mask = mask

            # compute final DII
            if self.distances_B is None:  # space B provided as features
                ranks_B = self._compute_rank_matrix(
                    batch_rows=self.data_B,
                    batch_columns=self.data_B,
                    periods=self.periods_B,
                )
            else:  # space B provided as distances
                ranks_B = self.distances_B.argsort(axis=1).argsort(axis=1)

            if mask is not None:
                ranks_B = ranks_B[mask].reshape((ranks_B.shape[0], -1))
                ranks_B = ranks_B.argsort(axis=1).argsort(axis=1) + 1

            # set k to keep same ratio k/N used during DII training
            k = int(
                jnp.ceil(
                    self.k_final
                    * self.batches_per_epoch
                    * (1 - 2 * discard_close_ind / self.ncolumns)
                )
            )
            imb_final = self._compute_final_diff_imbalance(
                params=self.params_final,
                batch_A_rows=self.data_A,
                batch_A_columns=self.data_A,
                batch_B_ranks=ranks_B,
                k=k,
            )
            error_final = None

        self.imb_final = imb_final
        self.error_final = error_final

        return imb_final, error_final

    def forward_greedy_feature_selection(
        self,
        n_features_max=None,
        n_best=10,
        compute_error=False,
        ratio_rows_columns=1,
        seed=0,
        discard_close_ind=0,
    ):
        """Performs forward greedy feature selection using the Differentiable Information Imbalance.

        Starting with all individual features, the algorithm evaluates which single feature has
        the lowest DII. Then it combines the best n_best single features with each
        of the remaining features to find the best 2-feature combination. This process continues
        until n_features_max features are selected or all features are included.

        For each candidate feature set, the weights are optimized specifically for that subset.
        When mini-batches are used, the same random seed ensures consistent mini-batch sequences, and the
        same split of points along rows and columns of distance matrices if compute_error is True.

        Args:
            n_features_max (int): maximum number of features to select. If None, will select up to all features.
            n_best (int): number of best feature tuples to consider at each iteration. Default is 10.
            compute_error (bool): whether to compute error estimates for the DII. Default is False.
            ratio_rows_columns (float): ratio between the number of points along rows and columns when
                computing the DII. Only used when compute_error is True. Default is 1.
            seed (int): seed for random number generation. Default is 0.
            discard_close_ind (int): index to discard close points when computing the DII. Default is 0.

        Returns:
            best_feature_sets (list): list of lists, where each sublist contains the indices of the selected
                features at each iteration.
            best_diis (list): list of DII values corresponding to each set of selected features.
            best_errors (list): list of error estimates for each DII value. Only meaningful if compute_error is True.
            best_weights_list (list): list of arrays containing the optimal weights for each set of selected features.
        """
        if self.l1_strength != 0.0:
            warnings.warn(f"The greedy search will run with l1 strength equal to 0.")
        assert (
            self.params_groups is None
        ), f"This method is not yet compatible with option 'params_groups'."
        n_features = self.nfeatures_A
        if n_features_max is None:
            n_features_max = n_features

        # Initialize lists to store results
        best_feature_sets = []
        best_diis = []
        best_errors = []
        best_weights_list = []

        ############################ First evaluate all single features ############################
        single_feature_diis = []
        single_feature_errors = []

        for feature in range(n_features):
            # Create mask for this single feature
            mask = jnp.zeros(n_features, dtype=bool)
            mask = mask.at[feature].set(True)

            # Initialize weights for training (only this feature is active)
            # Use the corresponding value from self.params_init for this feature
            params_init = jnp.where(mask, self.params_init, 0.0)

            # Create a copy of the current object for training
            dii_copy = DiffImbalance(
                data_A=self.data_A,
                data_B=self.data_B,
                distances_B=self.distances_B,
                periods_A=self.periods_A,
                periods_B=self.periods_B,
                seed=seed,
                num_epochs=self.num_epochs,
                batches_per_epoch=self.batches_per_epoch,
                l1_strength=0.0,
                point_adapt_lambda=self.point_adapt_lambda,
                k_init=self.k_init,
                k_final=self.k_final,
                lambda_factor=self.lambda_factor,
                params_init=params_init,
                optimizer_name=self.optimizer_name,
                learning_rate=self.learning_rate,
                learning_rate_decay=self.learning_rate_decay,
                num_points_rows=self.num_points_rows,
                gradient_clip_value=self.gradient_clip_value,
            )

            # Set initial parameters and train
            try:
                _, _ = dii_copy.train()
            except AssertionError as e:
                print(f"Training failed for feature [{feature}]: {str(e)}")
                print(f"Skipping feature [{feature}] and continuing...")
                single_feature_diis.append(
                    float("inf")
                )  # Use infinity as a large penalty
                single_feature_errors.append(None)
                continue

            # Compute DII on the full dataset
            if compute_error:
                dii_copy.return_final_dii(
                    compute_error=True,
                    ratio_rows_columns=ratio_rows_columns,
                    seed=seed,
                    discard_close_ind=discard_close_ind,
                )
                single_feature_diis.append(float(dii_copy.imb_final))
                single_feature_errors.append(float(dii_copy.error_final))
            else:
                dii_copy.return_final_dii(
                    compute_error=False,
                    ratio_rows_columns=None,
                    seed=seed,
                    discard_close_ind=discard_close_ind,
                )
                single_feature_diis.append(float(dii_copy.imb_final))
                single_feature_errors.append(None)

            print(f"Feature set = [{feature}], DII = {dii_copy.imb_final}\n")

        # Convert to numpy arrays for easier manipulation
        single_feature_diis = np.array(single_feature_diis)

        # Check if we have any valid features (not infinity)
        valid_features = np.isfinite(single_feature_diis)
        if not np.any(valid_features):
            print("ERROR: All single features failed during training!")
            return [], [], [], []

        # Select the best n_best single features (only from valid ones)
        valid_indices = np.where(valid_features)[0]
        valid_diis = single_feature_diis[valid_indices]
        n_best_actual = min(n_best, len(valid_indices))
        best_valid_indices = np.argsort(valid_diis)[:n_best_actual]
        selected_indices = valid_indices[best_valid_indices]

        # Convert indices to lists for consistent processing
        selected_features = [[idx] for idx in selected_indices]

        # Add the best single feature to results
        best_feature = selected_features[0]
        best_feature_sets.append(best_feature)
        best_diis.append(single_feature_diis[selected_indices[0]])

        # Store the optimal weights for the best single feature
        best_weights = np.zeros(n_features)
        best_weights[best_feature[0]] = self.params_init[
            best_feature[0]
        ]  # Inherit from parent class

        # Add to weights list
        best_weights_list.append(best_weights)

        if compute_error:
            best_errors.append(single_feature_errors[selected_indices[0]])
        else:
            best_errors.append(None)

        # Print the best single feature information
        print("------------------------------------------------")
        print(f"Best single feature: [{best_feature[0]}]")
        print(f"\tDII: {single_feature_diis[selected_indices[0]]}")
        print(f"\tOptimal weights: {best_weights}")
        print(f"Selected {n_best_actual} best candidates for next iteration")
        print("------------------------------------------------")

        # Get all features as a list
        all_features = list(range(n_features))

        ############################ Greedy loop over n-tuples (n>1) ############################
        while len(best_feature_sets[-1]) < min(n_features_max, n_features):
            candidate_features = []
            candidate_diis = []
            candidate_errors = []

            # Generate candidate feature sets by combining selected features with remaining features
            for selected_set in selected_features:
                for feature in all_features:
                    if feature not in selected_set:
                        # Create a new candidate set by adding this feature
                        candidate_set = selected_set + [feature]
                        candidate_set.sort()  # Sort for consistent comparison

                        # Skip if this set has already been evaluated
                        if candidate_set in candidate_features:
                            continue

                        candidate_features.append(candidate_set)

                        # Create mask for this candidate set
                        mask = jnp.zeros(n_features, dtype=bool)
                        mask = mask.at[jnp.array(candidate_set)].set(True)

                        # Initialize weights for training: inherit from parent class
                        params_init = jnp.where(mask, self.params_init, 0.0)

                        # Create a copy of the current object for training
                        dii_copy = DiffImbalance(
                            data_A=self.data_A,
                            data_B=self.data_B,
                            distances_B=self.distances_B,
                            periods_A=self.periods_A,
                            periods_B=self.periods_B,
                            seed=seed
                            + len(candidate_features),  # Ensure reproducibility
                            num_epochs=self.num_epochs,
                            batches_per_epoch=self.batches_per_epoch,
                            l1_strength=0.0,
                            point_adapt_lambda=self.point_adapt_lambda,
                            k_init=self.k_init,
                            k_final=self.k_final,
                            lambda_factor=self.lambda_factor,
                            params_init=params_init,
                            optimizer_name=self.optimizer_name,
                            learning_rate=self.learning_rate,
                            learning_rate_decay=self.learning_rate_decay,
                            num_points_rows=self.num_points_rows,
                            gradient_clip_value=self.gradient_clip_value,
                        )

                        # Set initial parameters and train
                        try:
                            _, _ = dii_copy.train()
                        except AssertionError as e:
                            print(
                                f"Training failed for feature set {candidate_set}: {str(e)}"
                            )
                            print(
                                f"Skipping feature set {candidate_set} and continuing..."
                            )
                            candidate_diis.append(
                                float("inf")
                            )  # Use infinity as a large penalty
                            candidate_errors.append(None)
                            continue

                        # Compute DII on the full dataset
                        if compute_error:
                            dii_copy.return_final_dii(
                                compute_error=True,
                                ratio_rows_columns=ratio_rows_columns,
                                seed=seed,
                                discard_close_ind=discard_close_ind,
                            )
                            candidate_diis.append(float(dii_copy.imb_final))
                            candidate_errors.append(float(dii_copy.error_final))
                        else:
                            dii_copy.return_final_dii(
                                compute_error=False,
                                ratio_rows_columns=None,
                                seed=seed,
                                discard_close_ind=discard_close_ind,
                            )
                            candidate_diis.append(float(dii_copy.imb_final))
                            candidate_errors.append(None)

                        print(
                            f"Feature set = {candidate_set}, DII = {dii_copy.imb_final}\n"
                        )

            # Convert to numpy arrays for easier manipulation
            candidate_diis = np.array(candidate_diis)

            if not candidate_features:  # No more features to add
                break

            # Check if we have any valid candidates (not infinity)
            valid_candidates = np.isfinite(candidate_diis)
            if not np.any(valid_candidates):
                print("ERROR: All candidate feature sets failed during training!")
                break

            # Select the best n_best candidates for the next iteration (only from valid ones)
            valid_indices = np.where(valid_candidates)[0]
            valid_diis = candidate_diis[valid_indices]
            n_best_actual = min(n_best, len(valid_indices))
            best_valid_indices = np.argsort(valid_diis)[:n_best_actual]
            best_indices = valid_indices[best_valid_indices]
            selected_features = [candidate_features[i] for i in best_indices]

            # Print the best feature set information
            best_idx = best_indices[0]

            # Add the best new set to results
            best_feature_sets.append(candidate_features[best_idx])
            best_diis.append(candidate_diis[best_idx])
            if compute_error:
                candidate_errors = np.array(candidate_errors)
                best_errors.append(candidate_errors[best_idx])
            else:
                best_errors.append(None)

            # Create a copy of DiffImbalance to get the optimal weights for the best feature set
            # (not saved before to avoid memory problems for large data sets)
            mask = jnp.zeros(n_features, dtype=bool)
            mask = mask.at[jnp.array(candidate_features[best_idx])].set(True)
            params_init = jnp.where(mask, self.params_init, 0.0)

            dii_copy = DiffImbalance(
                data_A=self.data_A,
                data_B=self.data_B,
                distances_B=self.distances_B,
                periods_A=self.periods_A,
                periods_B=self.periods_B,
                seed=seed,
                num_epochs=self.num_epochs,
                batches_per_epoch=self.batches_per_epoch,
                l1_strength=0.0,
                point_adapt_lambda=self.point_adapt_lambda,
                k_init=self.k_init,
                k_final=self.k_final,
                lambda_factor=self.lambda_factor,
                params_init=params_init,
                optimizer_name=self.optimizer_name,
                learning_rate=self.learning_rate,
                learning_rate_decay=self.learning_rate_decay,
                num_points_rows=self.num_points_rows,
                gradient_clip_value=self.gradient_clip_value,
            )

            # Set initial parameters and train
            try:
                _, _ = dii_copy.train()
                # Print and store optimal weights
                print(
                    f"\nOptimal weights for feature set {candidate_features[best_idx]}: {dii_copy.params_final}\n"
                )
                # Save optimal weights
                best_weights = np.array(dii_copy.params_final)
            except AssertionError as e:
                print(
                    f"Training failed for best feature set {candidate_features[best_idx]}: {str(e)}"
                )
                print(f"Using zero weights for this iteration...")
                best_weights = np.zeros(n_features)

            best_weights_list.append(best_weights)

            # Print the best n-tuple information
            print("------------------------------------------------")
            print(
                f"Best {len(best_feature_sets[-1])}-tuple: {candidate_features[best_idx]}"
            )
            print(f"\tDII: {candidate_diis[best_idx]}")
            print(f"\tOptimal weights: {best_weights}")
            print(f"Selected {n_best_actual} best candidates for next iteration")
            print("------------------------------------------------")

            # Stop if we've reached the maximum number of features
            if len(best_feature_sets[-1]) == n_features:
                break

        return best_feature_sets, best_diis, best_errors, best_weights_list

    def backward_greedy_feature_selection(
        self,
        n_features_min=1,
        n_best=10,
        compute_error=False,
        ratio_rows_columns=1,
        seed=0,
        discard_close_ind=0,
    ):
        """Performs backward greedy feature selection using the Differentiable Information Imbalance.

        Starting with all features, the algorithm progressively removes the least informative features
        one at a time, until either no features are left or n_features_min is reached.
        For each iteration, the algorithm selects the n_best feature sets with the lowest DII values
        for consideration in the next round.
        The method should be called after calling the train() method, which performs the first optimization.

        For each candidate feature set, the weights are optimized specifically for that subset.
        When mini-batches are used, the same random seed ensures consistent mini-batch sequences, and the
        same split of points along rows and columns of distance matrices if compute_error is True.

        Args:
            n_features_min (int): minimum number of features to select. Default is 1.
            n_best (int): number of best feature tuples to consider at each iteration. Default is 10.
            compute_error (bool): whether to compute error estimates for the DII. Default is False.
            ratio_rows_columns (float): ratio between the number of points along rows and columns when
                computing the DII. Only used when compute_error is True. Default is 1.
            seed (int): seed for random number generation. Default is 0.
            discard_close_ind (int): index to discard close points when computing the DII. Default is 0.

        Returns:
            feature_sets (list): list of lists, where each sublist contains the indices of the selected
                features at each iteration.
            diis (list): list of DII values corresponding to each set of selected features.
            errors (list): list of error estimates for each DII value. Only meaningful if compute_error is True.
            best_weights_list (list): list of arrays containing the optimal weights for each set of selected features.
        """
        if self.l1_strength != 0.0:
            warnings.warn(f"The greedy search will run with l1 strength equal to 0.")
        assert (
            self.params_groups is None
        ), f"This method is not yet compatible with option 'params_groups'."
        assert self.params_final is not None, "First call the train() method!"

        n_features = self.nfeatures_A

        # Initialize lists to store results
        feature_sets = []
        diis = []
        errors = []
        best_weights_list = []

        # Start with all features and use the original trained weights
        current_features = [list(range(n_features))]

        ############################ First evaluate all features together ############################
        if compute_error:
            self.return_final_dii(
                compute_error=True,
                ratio_rows_columns=ratio_rows_columns,
                seed=seed,
                discard_close_ind=discard_close_ind,
            )
            diis.append(float(self.imb_final))
            errors.append(float(self.error_final))
        else:
            self.return_final_dii(
                compute_error=False,
                ratio_rows_columns=None,  # Set to None when compute_error is False
                seed=seed,
                discard_close_ind=discard_close_ind,
            )
            diis.append(float(self.imb_final))
            errors.append(None)

        # Print all-feature information
        print("------------------------------------------------")
        print(f"All features: {current_features}")
        print(f"\tDII: {self.imb_final}")
        print(f"\tOptimal weights: {self.params_final}")
        print("------------------------------------------------")

        feature_sets.append(current_features[0].copy())
        best_weights_list.append(self.params_final)

        ############################ Greedy loop over n-tuples (n<D) ############################
        while feature_sets[-1] and len(feature_sets[-1]) > n_features_min:
            candidate_diis = []
            candidate_errors = []
            candidate_features = []

            # Generate candidates by removing one feature from each of the current best feature sets
            for selected_set in current_features:
                if len(selected_set) <= n_features_min:
                    # Skip sets that are already at minimum size
                    continue

                for i, feature in enumerate(selected_set):
                    # Create candidate feature set by removing this feature
                    candidate_set = selected_set.copy()
                    candidate_set.pop(i)

                    # Sort the candidate set for consistent comparison
                    candidate_set.sort()

                    # Skip if this set has already been evaluated
                    if candidate_set in candidate_features:
                        continue

                    candidate_features.append(candidate_set)

                    # Create mask for this candidate set
                    mask = jnp.zeros(n_features, dtype=bool)
                    mask = mask.at[jnp.array(candidate_set)].set(True)

                    # Initialize weights for training: inherit from parent class
                    params_init = jnp.where(mask, self.params_init, 0.0)

                    # Reset the random seed for consistent mini-batch sequence
                    training_seed = seed + len(candidate_features)

                    # Create a copy of the current object for training
                    dii_copy = DiffImbalance(
                        data_A=self.data_A,
                        data_B=self.data_B,
                        distances_B=self.distances_B,
                        periods_A=self.periods_A,
                        periods_B=self.periods_B,
                        seed=training_seed,
                        num_epochs=self.num_epochs,
                        batches_per_epoch=self.batches_per_epoch,
                        l1_strength=0.0,
                        point_adapt_lambda=self.point_adapt_lambda,
                        k_init=self.k_init,
                        k_final=self.k_final,
                        lambda_factor=self.lambda_factor,
                        params_init=params_init,
                        params_groups=None,
                        optimizer_name=self.optimizer_name,
                        learning_rate=self.learning_rate,
                        learning_rate_decay=self.learning_rate_decay,
                        num_points_rows=self.num_points_rows,
                        gradient_clip_value=self.gradient_clip_value,
                    )

                    # Set initial parameters and train
                    try:
                        _, _ = dii_copy.train()
                        # Store the trained weights
                        trained_weights = dii_copy.params_final
                    except AssertionError as e:
                        print(
                            f"Training failed for feature set {candidate_set}: {str(e)}"
                        )
                        print(f"Skipping feature set {candidate_set} and continuing...")
                        candidate_diis.append(
                            float("inf")
                        )  # Use infinity as a large penalty
                        candidate_errors.append(None)
                        continue

                    # Use return_final_dii to compute DII on the full dataset
                    dii_copy.params_final = trained_weights
                    if compute_error:
                        dii_copy.return_final_dii(
                            compute_error=True,
                            ratio_rows_columns=ratio_rows_columns,
                            seed=seed,
                            discard_close_ind=discard_close_ind,
                        )
                        candidate_diis.append(dii_copy.imb_final)
                        candidate_errors.append(dii_copy.error_final)
                    else:
                        dii_copy.return_final_dii(
                            compute_error=False,
                            ratio_rows_columns=None,  # Set to None when compute_error is False
                            seed=seed,
                            discard_close_ind=discard_close_ind,
                        )
                        candidate_diis.append(dii_copy.imb_final)
                        candidate_errors.append(None)

                    print(
                        f"Feature set = {candidate_set}, DII = {dii_copy.imb_final}\n"
                    )

            # Make sure we have candidates before proceeding
            if not candidate_features:
                print("No more candidates to evaluate, exiting backward search")
                break

            # Convert to numpy arrays for easier manipulation
            candidate_diis = np.array(candidate_diis)

            # Check if we have any valid candidates (not infinity)
            valid_candidates = np.isfinite(candidate_diis)
            if not np.any(valid_candidates):
                print("ERROR: All candidate feature sets failed during training!")
                break

            # Select the best n_best candidates (only from valid ones)
            valid_indices = np.where(valid_candidates)[0]
            valid_diis = candidate_diis[valid_indices]
            n_best_actual = min(n_best, len(valid_indices))
            best_valid_indices = np.argsort(valid_diis)[:n_best_actual]
            best_indices = valid_indices[best_valid_indices]

            # Update current features for the next iteration
            current_features = [candidate_features[i] for i in best_indices]

            # Select the best candidate (lowest DII)
            best_idx = best_indices[0]
            best_feature_set = candidate_features[best_idx]

            # Create a copy of DiffImbalance to get the optimal weights for the best feature set
            # (not saved before to avoid memory problems for large data sets)
            mask = jnp.zeros(n_features, dtype=bool)
            mask = mask.at[jnp.array(best_feature_set)].set(True)
            params_init = jnp.where(mask, self.params_init, 0.0)
            dii_copy = DiffImbalance(
                data_A=self.data_A,
                data_B=self.data_B,
                distances_B=self.distances_B,
                periods_A=self.periods_A,
                periods_B=self.periods_B,
                seed=seed,
                num_epochs=self.num_epochs,
                batches_per_epoch=self.batches_per_epoch,
                l1_strength=0.0,
                point_adapt_lambda=self.point_adapt_lambda,
                k_init=self.k_init,
                k_final=self.k_final,
                lambda_factor=self.lambda_factor,
                params_init=params_init,
                params_groups=None,
                optimizer_name=self.optimizer_name,
                learning_rate=self.learning_rate,
                learning_rate_decay=self.learning_rate_decay,
                num_points_rows=self.num_points_rows,
                gradient_clip_value=self.gradient_clip_value,
            )

            # Set initial parameters and train
            try:
                _, _ = dii_copy.train()
                # Save optimal weights
                best_weights = dii_copy.params_final
            except AssertionError as e:
                print(
                    f"Training failed for best feature set {best_feature_set}: {str(e)}"
                )
                print(f"Using zero weights for this iteration...")
                best_weights = np.zeros(n_features)

            best_weights_list.append(best_weights)

            # Store results
            feature_sets.append(best_feature_set.copy())
            diis.append(candidate_diis[best_idx])

            if compute_error:
                candidate_errors = np.array(candidate_errors)
                errors.append(candidate_errors[best_idx])
            else:
                errors.append(None)

            # Print the best n-tuple information
            print("------------------------------------------------")
            print(f"Best {len(best_feature_set)}-tuple: {candidate_features[best_idx]}")
            print(f"\tDII: {candidate_diis[best_idx]}")
            print(f"\tOptimal weights: {best_weights}")
            print(f"Selected {n_best_actual} best candidates for next iteration")
            print("------------------------------------------------")

        return feature_sets, diis, errors, best_weights_list
