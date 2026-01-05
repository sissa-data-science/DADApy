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
The *causal_graph* module contains the *CausalGraph* class, which inherits from the *DiffImbalance* class.

The code can be runned on gpu using the command
    jax.config.update('jax_platform_name', 'gpu') # set 'cpu' or 'gpu'
"""

import string
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from dadapy import DiffImbalance


class CausalGraph(DiffImbalance):
    """Constructs a community causal graph where variables are grouped into single nodes.

    Attributes:
        time_series (np.array(float)): array of shape (N_times,D), where N_times is the length of
            trajectory and D is the number of dynamical variables. The sampling time is supposed to
            be constant along the trajectory and for all the variables.
        coords_present (np.array(float)): array of shape (N_samples,D) containing the samples of the
            D-dimensional trajectory at time t=0; read only when time_series is None.
        coords_future (np.array(float)): array of shape (N_samples,D,n_lags) containing the samples of
            the D-dimensional trajectory at different future time lags; read only when time_series is None.
            If you want to test a single time lag, reshape the dataset with coords_future[:,:,np.newaxis].
        periods (np.ndarray(float)): array of shape (D,) containing the periods of the dynamical variables.
            The default is None, which means that the variables are treated as nonperiodic. If not all
            variables are periodic, the entries of the nonperiodic ones should be set to 0.
        standardize (bool): whether to standardize each of the D variables, dividing by its standard
            deviation along the trajectory or the provided samples. Default is True.
        seed (int): seed of JAX random generator.
    """

    def __init__(
        self,
        time_series=None,
        coords_present=None,
        coords_future=None,
        periods=None,
        standardize=True,
        seed=0,
    ):
        self.time_series = time_series
        self.coords_present = coords_present
        self.coords_future = coords_future
        self.standardize = standardize
        self.num_variables, self.periods = self._check_and_initialize_args(periods)
        self.seed = seed
        self.imbs_training = None
        self.weights_training = None
        self.weights_final = None
        self.imbs_final = None
        self.errors_final = None
        self.adj_matrix = None
        self.community_dictionary = None
        self.cov = None

    def _check_and_initialize_args(self, periods):
        """Checks input arguments to constructor of CausalGraph object."""
        num_variables = None
        periods = periods
        if (
            self.time_series is None
            and self.coords_present is not None
            and self.coords_future is not None
        ):
            assert len(self.coords_future.shape) == 3, (
                f"Coords_future has shape {self.coords_future.shape}, while the expected shape "
                + f"is (N_samples, D_features, n_lags).\nIf you want to test a single time lag, "
                + f"provide as input coords_future[:,:,np.newaxis]."
            )
            n_lags = self.coords_future.shape[2]
            assert self.coords_present.shape == self.coords_future.shape[:2], (
                "Arguments coords_present and coords_future should have shapes (N_samples, D_features) "
                + f"and (N_samples, D_features, n_lags),\n but the number of samples and/or the "
                + f"number of features do not match."
            )
            num_variables = self.coords_present.shape[1]
            if periods is not None:
                periods = np.ones(self.coords_present.shape[1]) * np.array(periods)
            if (
                self.standardize is False
                and (
                    np.std(self.coords_present, ddof=1, axis=0)
                    != np.ones(num_variables)
                ).any()
            ):
                warnings.warn(
                    f"The {num_variables} variables in 'coords_present' are not standardized."
                )
            if self.standardize is True:
                self.coords_present /= np.std(
                    self.coords_present, ddof=1, axis=0, keepdims=True
                )
        elif self.time_series is not None:
            if self.coords_present is not None or self.coords_future is not None:
                warnings.warn(
                    f"You passed the whole time series as input; the arguments coords_present and "
                    + f"coords_future will be ignored"
                )
            num_variables = self.time_series.shape[1]
            if periods is not None:
                periods = np.ones(time_series.shape[1]) * np.array(periods)
            if (
                self.standardize is False
                and (
                    np.std(self.time_series, ddof=1, axis=0) != np.ones(num_variables)
                ).any()
            ):
                warnings.warn(
                    f"Warning: the {num_variables} variables of the input time series are not standardized."
                )
            if self.standardize is True:
                self.time_series /= np.std(
                    self.time_series, ddof=1, axis=0, keepdims=True
                )
        return num_variables, periods

    def return_nn_indices(
        self,
        variables,
        num_samples,
        time_lags,
        embedding_dim=1,
        embedding_time=1,
        discard_close_ind=None,
    ):
        """Returns the indices of the nearest neighbor of each point.

        Args:
            variables (list, jnp.array(int)): array of the coordinates used to build the distance space (with weights 1).
            num_samples (int): number of samples harvested from the full time series.
            time_lags (list(int), np.array(int)): tested time lags between 'present' and 'future'.
            embedding_dim (int): dimension of the time-delay embedding vector built on each variable. Default is 1,
                which means the time-delay embeddings are not employed.
            embedding_time (int): lag between consecutive samples in the time-delay embedding vectors of each
                variable. Default is 1.
            discard_close_ind (int): defines the "close points" for which distances and ranks are not computed: for each
                point i, distances between i and points within [i-discard_close_ind, i+discard_close_ind] are discarded.

        Returns:
            nn_indices (np.array(float)): array of the nearest neighbors indices: nn_indices[i] is the index of the column
                with value 1 in the rank matrix.
        """
        assert (
            self.time_series is not None
        ), "Error: to call this method, provide the time series while initializing the CausalGraph class."

        assert num_samples <= self.time_series.shape[0] - max(time_lags), (
            f"Error: cannot extract {num_samples} samples from {self.time_series.shape[0]} initial samples, "
            + f"if the maximum time lag is {max(time_lags)}.\nChoose a value of num_samples such that "
            + f"num_samples < {self.time_series.shape[0]} - {max(time_lags)}"
        )
        indices_present = np.linspace(
            (embedding_dim - 1)
            * embedding_time,  # select times defining the ensemble of trajectories
            self.time_series.shape[0] - max(time_lags) - 1,
            num_samples,
            dtype=int,
        )
        indices_present = [
            indices_present - embedding_time * i for i in range(embedding_dim)
        ]
        coords_present = self.time_series[
            indices_present
        ]  # has shape (embedding_dim, num_samples, n_variables)
        coords_present = np.transpose(
            coords_present, axes=[1, 2, 0]
        )  # convert to shape (num_samples, n_variables, embedding_dim)
        dii = DiffImbalance(
            data_A=coords_present[:, variables].reshape(
                (num_samples, len(variables) * embedding_dim)
            ),
            data_B=coords_present[:, variables].reshape(
                (num_samples, len(variables) * embedding_dim)
            ),  # dummy argument
        )
        nn_indices = dii._return_nn_indices(discard_close_ind=discard_close_ind)
        return np.array(nn_indices)

    def optimize_present_to_future(
        self,
        num_samples,
        time_lags,
        embedding_dim_present=1,
        embedding_dim_future=1,
        embedding_time=1,
        target_variables="all",
        save_weights=False,
        num_epochs=200,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k_init=1,
        k_final=1,
        lambda_factor=0.1,
        params_init=None,
        optimizer_name="sgd",
        learning_rate=1e-2,
        learning_rate_decay=None,
        num_points_rows=None,
        compute_imb_final=False,
        compute_error=False,
        ratio_rows_columns=1,
        discard_close_ind=None,
        langevin_steps=0,
        early_stopping=0,
        return_covariance=False,
    ):
        """Iteratively optimizes the DII from the full space in the present to a target space in the future.

        Arguments 'num_samples', 'time_lags', 'embedding_dim_present', 'embedding_dim_future' and 'embedding_time'
        are read only when data are provided to the CausalGraph object through the argument 'time_series'.
        Arguments 'compute_error', 'ratio_rows_columns' and 'discard_close_ind' are only read when 'compute_imb_final'
        is set to True.

        Args:
            num_samples (int): number of samples harvested from the full time series, interpreted as
                independent initial conditions of the same dynamical process.
            time_lags (list(int), np.ndarray(int)): tested time lags between 'present' and 'future'.
            embedding_dim_present (int): dimension of the time-delay embedding vectors built in the present
                space (t=0, t=-1, ...). Default is 1, which means the time-delay embeddings are not employed.
            embedding_dim_future (int): dimension of the time-delay embedding vectors built in the space of
                the target variable (t=tau, t=tau-1, ...). Default is 1.
            embedding_time (int): lag between consecutive samples in the time-delay embedding vectors of each
                variable.  Default is 1.
            target_variables (str or list(int), np.array(int)): list or np.array of the target variables
                defining the distance space in the future. Default is "all", for which the optimization is
                iterated over all variables as target.
            save_weights (bool): whether to save or not the weights during training, rather than only the final
                weights. If True, weights are saved in the attribute 'weights_training' of the CausalGraph object,
                which is an array of shape (n_target_variables, n_time_lags, num_epochs+1, num_variables).
                Default is False.
            num_epochs (int): number of training epochs. Default is 200.
            batches_per_epoch (int): number of minibatches; must be a divisor of n_points. Each weight update is
                carried out by computing the DII gradient over n_points / batches_per_epoch points. Default is 1,
                which means that the gradient is computed over all the available points (batch GD).
            seed (int): seed of JAX random generator, default is 0. Different seeds determine different mini-batch
                partitions.
            l1_strength (float): strength of the L1 regularization (LASSO) term. Default is 0.
            point_adapt_lambda (bool): whether to use a global smoothing parameter lambda for the c_ij coefficients
                in the DII (if False), or a different parameter for each point (if True). Default is True.
            k_init (int): initial rank of neighbors used to set lambda. Ranks are defined starting from 1. If
                batches_per_epoch > 1, neighbors are recomputed within each mini-batch. Default is 1.
            k_final (int): final rank of neighbors used to set lambda. If batches_per_epoch > 1, neighbors are
                recomputed within each mini-batch. Default is 1.
            lambda_factor (float): factor defining the scale of lambda. Default is 0.1.
            params_init (np.array(float), jnp.array(float)): array of shape (n_features_A,) containing the initial
                values of the scaling weights to be optimized. If None, params_init is set to [0.1, 0.1, ..., 0.1].
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
            compute_imb_final (bool): whether to compute the final DII over the full data set, using the options
                specified by 'compute_error', 'ratio_rows_columns' and 'discard_close_ind'. Default is False, for
                which those arguments are ignored.
            compute_error (bool): whether to compute the final DII and its error by sampling different points along
                rows and columns of the distance matrix. If False, the final DII is computed using the same points
                along rows and columns, which does not allow for an error estimation. Default is True.
            ratio_rows_columns (float): only read when compute_error is True; defines the ratio between the number
                of points along rows (nrows) and along columns (ncolumns) of distance and rank matrices, in two groups
                randomly sampled.  In general, nrows and ncolumns are determined by solving the equations
                    nrows / ncolumns = ratio_rows_columns,
                    nrows + ncolumns = n_total_points.
                Default is 1, which means that both groups have n_points / 2 elements.
            discard_close_ind (int): given any point i, defines the "close" points (following the time ordering
                along axis=0 of 'time_series' or 'coords_present') that are known to be significantly correlated with i.
                If compute_error is True, "time-correlated" points are excluded by subsampling the data along axis=0
                with stride discard_close_ind + 1. If compute_error is False, distances between each point i and points
                within the time window [i-discard_close_ind, i+discard_close_ind] are discarded. Default is 0, for which
                no distances between points close in the time are discarded.
            langevin_steps (int): number of Langevin steps to perform after minimization.
            early_stopping (float): the threshold for the early stopping criterion. Default is 0, which means that no early stopping is performed.
            return_covariance (bool): whether to return the covariance matrix of the weights, obtained after Langevin dynamics

        Returns:
            weights_final (np.array(float)): array of shape (n_target_variables, n_time_lags, D) containing the
                D final scaling weights for each optimization, where D is the number of variables in the time series.
                If embedding_dim_present > 1, the shape is (n_target_variables, n_time_lags, D, embedding_dim_present).
                Also accessible as attribute of the CausalGraph object.
            imbs_training (np.array(float)): array of shape (n_target_variables, n_time_lags, num_epochs+1)
                containing the DII during the trainings. Also accessible as attribute of the CausalGraph object.
            imbs_final (np.array(float)): array of shape (n_target_variables, n_time_lags) containing the DII at
                the end of each training computed over the full data set. If 'compute_imb_final' is False, imbs_final
                is set to None. Also accessible as attribute of the CausalGraph object.
            errors_final (np.array(float)): array of shape (n_target_variables, n_time_lags) containing the errors
                of the DII at the end of each training, computed over the full data set. If 'compute_imb_final' is False,
                or if 'compute_imb_final' is True and 'compute_error' is False, errors_final is set to None. Also
                accessible as attribute of the CausalGraph object.
        """
        coords_present = None
        if self.time_series is not None:
            assert num_samples <= self.time_series.shape[0] - max(time_lags), (
                f"Error: cannot extract {num_samples} samples from {self.time_series.shape[0]} initial "
                + f"samples, if the maximum time lag is {np.max(time_lags)}.\nChoose a smaller value of "
                + f"num_samples."
            )

            t0s = np.linspace(
                (max(embedding_dim_present, embedding_dim_future) - 1)
                * embedding_time,  # select times defining the ensemble of trajectories
                self.time_series.shape[0] - max(time_lags) - 1,
                num_samples,
                dtype=int,
            )
            indices_present = np.array(
                [t0s - embedding_time * i for i in range(embedding_dim_present)]
            )
            coords_present = self.time_series[
                indices_present
            ]  # has shape (embedding_dim_present, num_samples, n_variables)
            coords_present = np.transpose(
                coords_present, axes=[1, 2, 0]
            )  # convert to shape (num_samples, n_variables, embedding_dim_present)
            coords_present = coords_present.reshape(
                (num_samples, self.num_variables * embedding_dim_present)
            )
        elif self.coords_present is not None:
            if num_samples is not None:
                warninings.warn(
                    f"Argument 'num_samples' will be ignored, as you already provided the independent "
                    + f"initial conditions through arguments 'coords_present' and 'coords_future'.\n "
                    + f"To suppress this warning, set 'num_samples' to None."
                )
            if time_lags is not None:
                warninings.warn(
                    f"Argument 'time_lags' will be ignored, as the samples at different time lags t=tau "
                    + f"are already read from the last dimension of 'coords_future'.\n "
                    + f"To suppress this warning, set 'time_lags' to None."
                )
            num_samples = self.coords_present.shape[0]
            time_lags = np.arange(1, self.coords_future.shape[2] + 1)
            coords_present = self.coords_present
        else:
            print(
                "To call this method, provide either a time series or directly the present and future samples "
                + f"while initializing the CausalGraph class."
            )

        if target_variables == "all":
            target_variables = np.arange(self.num_variables)

        # initialize output variables
        imbs_temp = np.zeros((len(target_variables), len(time_lags), num_epochs + 1))
        imbs_training = np.zeros(
            (len(target_variables), len(time_lags), num_epochs + langevin_steps + 1)
        )
        if return_covariance:
            covariance_matrix = np.zeros(
                (
                    len(target_variables),
                    len(time_lags),
                    self.num_variables * embedding_dim_present,
                    self.num_variables * embedding_dim_present,
                )
            )

        if embedding_dim_present == 1:
            weights_final = np.zeros(
                (len(target_variables), len(time_lags), self.num_variables)
            )
            if save_weights is True:
                weights_training = np.zeros(
                    (
                        len(target_variables),
                        len(time_lags),
                        num_epochs + langevin_steps + 1,
                        self.num_variables,
                    )
                )
        elif embedding_dim_present > 1:
            weights_final = np.zeros(
                (
                    len(target_variables),
                    len(time_lags),
                    self.num_variables,
                    embedding_dim_present,
                )
            )
            if save_weights is True:
                weights_training = np.zeros(
                    (
                        len(target_variables),
                        len(time_lags),
                        num_epochs + langevin_steps + 1,
                        self.num_variables,
                        embedding_dim_present,
                    )
                )
        imbs_final = None
        errors_final = None
        if compute_imb_final:
            imbs_final = np.zeros((len(target_variables), len(time_lags)))
            if compute_error:
                errors_final = np.zeros((len(target_variables), len(time_lags)))

        # loop over target variables and time lags
        for i_var, target_var in enumerate(target_variables):
            for j_tau, tau in enumerate(time_lags):
                indices_future = (
                    np.array(
                        [t0s - embedding_time * i for i in range(embedding_dim_future)]
                    )
                    + tau
                )

                if self.time_series is not None:
                    coords_future = self.time_series[
                        indices_future, target_var
                    ]  # has shape (embedding_dim_future, num_samples)
                    coords_future = np.transpose(
                        coords_future, axes=[1, 0]
                    )  # convert to shape (num_samples, embedding_dim_future)
                else:
                    coords_future = self.coords_future[:, :, j_tau]

                dii = DiffImbalance(
                    data_A=coords_present,
                    data_B=coords_future,
                    periods_A=self.periods,
                    periods_B=(
                        None if self.periods is None else self.periods[target_var]
                    ),
                    seed=self.seed,
                    num_epochs=num_epochs,
                    batches_per_epoch=batches_per_epoch,
                    l1_strength=l1_strength,
                    point_adapt_lambda=point_adapt_lambda,
                    k_init=k_init,
                    k_final=k_final,
                    lambda_factor=lambda_factor,
                    params_init=params_init,
                    optimizer_name=optimizer_name,
                    learning_rate=learning_rate,
                    learning_rate_decay=learning_rate_decay,
                    num_points_rows=num_points_rows,
                    early_stopping=early_stopping,
                )
                weights_temp, imbs_temp[i_var, j_tau] = dii.train(
                    bar_label=f"target_var={target_var}, tau={tau}"
                )

                # compute final DII and its error
                if compute_imb_final:
                    imb, err = dii.return_final_dii(
                        compute_error=compute_error,
                        ratio_rows_columns=ratio_rows_columns,
                        seed=self.seed,
                        discard_close_ind=discard_close_ind,
                    )
                    imbs_final[i_var, j_tau] = imb
                    if compute_error:
                        errors_final[i_var, j_tau] = dii.error_final

                if langevin_steps != 0:
                    weights_langevin, imbs_langevin = dii.langevin(
                        weights_temp[-1],
                        n_epochs=langevin_steps,
                        batch_size=int(len(coords_present) / batches_per_epoch),
                        noise=np.mean(weights_temp[-1]) * 1e-1,
                    )
                    weights_temp = np.concatenate(
                        [weights_temp, weights_langevin[1:]], axis=0
                    )
                    imbs_training[i_var, j_tau] = np.concatenate(
                        [imbs_temp[i_var, j_tau], imbs_langevin[1:]]
                    )
                    if return_covariance:
                        covariance_matrix[i_var, j_tau] = np.cov(
                            weights_langevin[::10], rowvar=False
                        ) / len(weights_langevin[::10])

                # save weights
                if embedding_dim_present == 1:
                    weights_final[i_var, j_tau] = np.abs(
                        np.mean(weights_temp[-langevin_steps - 1 :], axis=0)
                    )
                    if save_weights is True:
                        weights_training[i_var, j_tau] = weights_temp.reshape(
                            (num_epochs + 1 + langevin_steps, self.num_variables)
                        )
                elif embedding_dim_present > 1:
                    weights_final[i_var, j_tau] = np.abs(
                        np.mean(weights_temp[-langevin_steps - 1 :], axis=0)
                    ).reshape((self.num_variables, embedding_dim_present))
                    if save_weights is True:
                        weights_training[i_var, j_tau] = weights_temp.reshape(
                            (
                                num_epochs + 1 + langevin_steps + langevin_steps,
                                self.num_variables,
                                embedding_dim_present,
                            )
                        )

        self.weights_final = weights_final
        self.imbs_training = imbs_training
        if save_weights:
            self.weights_training = weights_training
        self.imbs_final = imbs_final
        self.errors_final = errors_final

        if return_covariance == True:
            self.cov = covariance_matrix
            return (
                weights_final,
                imbs_training,
                imbs_final,
                errors_final,
                covariance_matrix,
            )
        return weights_final, imbs_training, imbs_final, errors_final

    def compute_adj_matrix(self, weights, threshold=1e-1):
        """Computes the adjacency matrix from the optimal weights returned by optimize_present_to_future.

        As a preliminary step before applying the threshold, the maximum weight over the tested time lags is
        taken for each pair X_i(0) -> X_j(tau) (i,j=1,...,D). If the weights are referred to time-delay
        embeddings, the maximum is also taken along the embedding dimension.

        Args:
            weights (np.ndarray(float)): array of shape (D, n_time_lags, D) containing the optimal scaling
                weights produced by optimize_present_to_future with the option target_variables="all". If the
                optimization was carried out with embedding_dim_present > 1, the array should have an additional
                dimension, i.e. shape (D, n_time_lags, D, embedding_dim_present).
            threshold (float): value of the threshold used to construct the adjacency matrix. If a weight is
                smaller than the threshold the corresponding entry in the adjacency matrix is set to 0, otherwise
                it is set to 1.

        Returns:
            adj_matrix (np.ndarray(float)): array of shape (D,D) defining the adjacency matrix of a directed
                graph, where each arrow defines a direct or indirect link. Also accessible as attribute of the
                CausalGraph object.
        """
        assert weights is not None, (
            f"To call this method, provide the weights obtained with the method optimize_present_to_future, "
            + f"with the option target_variables='all'"
        )
        assert len(weights.shape) == 3 or len(weights.shape) == 4, (
            "The array of weight must have shape (D,n_time_lags,D), or (D,n_time_lags,D,embedding_dim_present). "
            + f"If you are testing a single time lag, reshape this input as weights[:,np.newaxis,:]."
        )
        assert weights.shape[0] == weights.shape[2], (
            "The array of weight must have shape (D,n_time_lags,D), or (D,n_time_lags,D,embedding_dim_present), "
            + f"where D is the number of variables."
        )
        if len(weights.shape) == 3:
            weights_max = np.max(weights, axis=1)  # maximum over all tested time lags
        elif len(weights.shape) == 4:
            weights_max = np.max(
                weights, axis=(1, 3)
            )  # maximum over all tested time lags and embedding components

        D = weights.shape[0]
        adj_matrix = np.zeros((D, D))
        adj_matrix[weights_max.T > threshold] = 1  # apply threshold
        self.adj_matrix = adj_matrix
        return adj_matrix

    def _ancestors(self, adj_matrix):
        """Finds ancestors of each node in the directed graph described by the input adjacency matrix.

        Args:
            adj_matrix (np.ndarray(float)): binary matrix of shape (D,D) defining the links of a directed
                graph.

        Returns:
            auto_sets (list): list of lists, such that auto_sets[i] is a list containing the indices of all
                ancestors of node i in the graph
        """
        G = nx.DiGraph(adj_matrix)
        auto_sets = []
        for var in np.arange(adj_matrix.shape[0]):
            auto_sets.append(sorted(nx.ancestors(G, var) | {var}))
        return auto_sets

    def find_communities(self, adj_matrix):
        """Finds dynamical communities, i.e. groups of variables defining single nodes in the community causal graph.

        Args:
            adj_matrix (np.ndarray(float)): binary matrix of shape (D,D) defining the links of a directed
                graph with D nodes.

        Returns:
            community_dictionary (dict): dictionary with pairs (comm_id, level) as keys and lists containing
                the indices of the variables in each community as values. 'comm_id' is an integer number
                identifying the dynamical community, while 'level' is an integer identifying the step of the
                algorithm at which the community is identified, namely its level of autonomy. The keys are sorted
                from the smallest to the largest level. Both 'comm_id' and 'level' start from 0.
        """
        auto_sets = self._ancestors(adj_matrix)
        # re-order minimal autonomous sets from smallest to largest size
        sizes_auto_sets = [len(auto_set) for auto_set in auto_sets]
        auto_sets = [auto_sets[i_sorted] for i_sorted in np.argsort(sizes_auto_sets)]

        comm_index = 0
        level_index = 0
        variables_assigned = []
        community_dictionary = {}
        auto_sets_left = auto_sets.copy()

        while len(auto_sets_left) != 0:
            nvariables_left = len(auto_sets_left)
            sets_sizes = [
                len(auto_sets_left[i_comm]) for i_comm in range(nvariables_left)
            ]
            smallest_set_index = np.argmin(sets_sizes)
            community_dictionary[comm_index, level_index] = auto_sets_left[
                smallest_set_index
            ]
            variables_assigned.extend(auto_sets_left[smallest_set_index])
            auto_sets_left.pop(
                smallest_set_index
            )  # delete set from list of autonomous sets
            comm_index += 1

            parallel_communities = (
                0  # number of communities found to be autonomous at same level
            )
            auto_sets_left_temp = auto_sets_left.copy()
            for try_set_index, try_set in enumerate(auto_sets_left):
                intersection = set(variables_assigned).intersection(try_set)
                if intersection == set():
                    community_dictionary[comm_index, level_index] = auto_sets_left[
                        try_set_index
                    ]
                    variables_assigned.extend(auto_sets_left[try_set_index])
                    auto_sets_left_temp.pop(
                        try_set_index - parallel_communities
                    )  # delete set from list of minimal autonomous subsets
                    comm_index += 1
                    parallel_communities += 1
            auto_sets_left = auto_sets_left_temp.copy()

            for left_set_index, left_set in enumerate(auto_sets_left):
                auto_sets_left[left_set_index] = list(
                    set(left_set).difference(variables_assigned)
                )
            # delete empty lists
            auto_sets_left = [
                set_left for set_left in auto_sets_left if len(set_left) > 0
            ]
            level_index += 1

        self.community_dictionary = community_dictionary
        return community_dictionary

    def community_graph_visualization(
        self,
        community_dictionary,
        adj_matrix,
        type="community",
        savefig_name=None,
        **kwargs,
    ):
        """Shows a visual representation of the dynamical communities on a graph.

        This function makes use of the library networkx (https://networkx.org/documentation/stable/index.html)

        Args:
            community_dictionary (dict): dictionary with pairs (comm_id, level) as keys and lists containing
                the indices of the variables in each dynamical community as values.
            adj_matrix (np.array(float)): matrix of shape (D,D) defining the links between the variables
                after thresholding the matrix of the optimized weights.
            type (str): type of graph where the dynamical communities are represented, possible options are
                "community" (default) and "all-variable". If "community", a community causal graph where
                each node represents a community is shown. If "all-variable", communities are represented
                with different colors in a graph with all the original D variables in the time series.
            savefig_name (str): path at which the picture of the final graph is saved in pdf format. If
                None (default), the figure is not saved.
            **kwargs: customizable arguments used by the networkx library. If type="all-variable", these
                include: 'scale','k1' and 'k2', 'cmap', 'width' and 'arrowsize'. If type="community", the
                possible arguments are: 'node_color', 'node_size', 'width', 'arrowstyle', 'arrowsize'.

        Returns:
            G (nx.diGraph object): final causal graph.
        """
        assert (
            adj_matrix is not None
        ), "Provide as intput the adjacency matrix computed with the method compute_adj_matrix"
        assert (
            community_dictionary is not None
        ), "Provide as intput the community dictionary computed with the method find_communities"

        if type == "all-variable":
            G_ = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

            features = {}
            metafeatures = {}

            for element in community_dictionary.items():
                for variable in element[1]:
                    features.update({variable: element[0][0]})
                    metafeatures.update({variable: element[0][1]})

            communities = [
                set([el for el, pos in features.items() if pos == k])
                for k in set(features.values())
            ]
            metacommunities = [
                set([el for el, pos in metafeatures.items() if pos == k])
                for k in set(metafeatures.values())
            ]

            assert (
                len(communities) != 1
            ), f"Only one community is present. Try plotting with a standard function of networkx."

            G = nx.DiGraph()
            for comm in communities:
                G.add_node(str(list(comm)))
            for i in range(len(communities)):
                present = list(communities[i])
                time = metafeatures[present[0]]
                if time < len(metacommunities) - 1:
                    for step in range(1, len(metacommunities) - time):
                        future = list(
                            metacommunities[
                                metafeatures[list(communities[i])[0]] + step
                            ]
                        )
                        connections = np.where(adj_matrix[np.ix_(present, future)] != 0)
                        for j in range(len(connections[0])):
                            looking = future[connections[1][j]]
                            final = communities[
                                np.where(
                                    [
                                        looking in communities[i]
                                        for i in range(len(communities))
                                    ]
                                )[0][0]
                            ]
                            G.add_edge(str(present), str(list(final)))

            iter = [el for el in G.edges]
            for edge in iter:
                if (
                    sum(
                        1
                        for _ in nx.all_simple_paths(G, source=edge[0], target=edge[1])
                    )
                    > 1
                ):
                    G.remove_edge(edge[0], edge[1])

            options = {
                "scale": 0.1,
                "k1": 1,
                "k2": 2,
                "cmap": plt.cm.Blues,
                "width": 1,
                "arrowsize": 12,
            }
            options.update(kwargs)

            # Compute positions for the node clusters as if they were themselves nodes in a
            # supergraph using a larger scale factor
            superpos = nx.spring_layout(G, k=options["k1"], seed=429)

            # Use the "supernode" positions as the center of each node cluster
            centers = list(superpos.values())
            pos = {}
            for center, comm in zip(centers, communities):
                pos.update(
                    nx.spring_layout(
                        nx.subgraph(G_, comm),
                        scale=options["scale"],
                        k=options["k2"],
                        center=center,
                        seed=1430,
                    )
                )

            nx.draw(
                G_,
                pos=pos,
                node_color=[metafeatures[i] for i in range(len(adj_matrix))],
                cmap=options["cmap"],
                with_labels=True,
                width=options["width"],
                arrowsize=options["arrowsize"],
            )
            if savefig_name is not None:
                plt.savefig(savefig_name, dpi=300, bbox_inches="tight")
            plt.show()
            return G

        elif type == "community":
            # construct graph
            G = nx.DiGraph()
            keys = list(community_dictionary.keys())
            values = list(community_dictionary.values())
            alphabet_string = list(string.ascii_uppercase)
            community_names = {
                tuple(community): alphabet_string[i]
                for i, community in enumerate(values)
            }
            from_names_to_communities = {
                community_names[key]: key for key in community_names.keys()
            }

            # convert communities into names and add them to graph as nodes
            for community, key in zip(community_names, keys):
                community_name = community_names[tuple(community)]
                G.add_node(str(community_name))
                print(
                    f"Community {community_name} ({len(community)} variables, level {key[1]}): {community}"
                )

            # dictionary with keys: (order_idx) and values: list of communities at that order (list of list)
            communities_orders = {key[1]: [] for key in keys}
            for community_idx, order_idx in keys:
                communities_orders[order_idx].append(
                    community_dictionary[community_idx, order_idx]
                )

            # draw edges
            for community_effect_idx, order_idx in keys:
                if order_idx > 0:
                    # for each putative effect community at order >=1...
                    for community_effect in communities_orders[order_idx]:
                        community_name_effect = community_names[tuple(community_effect)]
                        # ...loop over all putative causal communities at order -1
                        previous_order = order_idx - 1
                        for community_cause in communities_orders[previous_order]:
                            community_name_cause = community_names[
                                tuple(community_cause)
                            ]
                            # ...loop over all variables in each putative causal community
                            for variable_cause in community_cause:
                                # ...and draw an edge if at least a link is found
                                if adj_matrix[variable_cause, community_effect].any():
                                    G.add_edges_from(
                                        [
                                            (
                                                str(community_name_cause),
                                                str(community_name_effect),
                                            )
                                        ]
                                    )
                                    break
            # show graph
            options = {
                "node_color": "gray",
                "node_size": 3000,
                "width": 3,
                "arrowstyle": "-|>",
                "arrowsize": 12,
            }
            options.update(kwargs)
            nx.draw_circular(G, arrows=True, with_labels=True, **options)
            if savefig_name is not None:
                plt.savefig(savefig_name, dpi=300, bbox_inches="tight")
            plt.show()

            # return networkx object
            return G
