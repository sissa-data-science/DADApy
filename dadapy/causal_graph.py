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
    jax.config.update('jax_platforms', 'gpu') # set 'cpu' or 'gpu'
"""

import itertools
import string
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from dadapy.diff_imbalance import DiffImbalance


def symbol_generator():
    """Generate the alphanumeric strings used to label the dynamical communities."""
    yield from string.ascii_uppercase
    for i in itertools.count(1):
        for c in string.ascii_uppercase:
            yield f"{c}{i}"


class CausalGraph(DiffImbalance):
    """Constructs a community causal graph where variables are grouped into single nodes.

    Attributes:
        time_series (np.array(float)): array of shape (N_times,D), where N_times is the length of
            trajectory and D is the number of dynamical variables. The sampling time is supposed to
            be constant along the trajectory and for all the variables.
        time_series_ensemble (np.array(float)): array of shape (N_trajs,N_times,D) containing N_trajs
            independent trajectories of the same dynamical process, sampled at the same N_times times; read
            only when time_series is None. Each trajectory provides one sample, taken at the same time t=0 for
            all the trajectories: t=0 is the first time along axis=1 or, if time-delay embeddings are employed,
            the time (max(embedding_dim_present, embedding_dim_future) - 1) * embedding_time, so that the
            embedding times before t=0 are available.
        periods (np.ndarray(float)): array of shape (D,) containing the periods of the dynamical variables.
            The default is None, which means that the variables are treated as nonperiodic. If not all
            variables are periodic, the entries of the nonperiodic ones should be set to 0.
        standardize (bool): whether to standardize each of the D variables, dividing by its standard
            deviation along the trajectory (or over all trajectories and times, for time_series_ensemble).
            Default is True.
        seed (int): seed of JAX random generator.
    """

    def __init__(
        self,
        time_series=None,
        time_series_ensemble=None,
        periods=None,
        standardize=True,
        seed=0,
    ):
        """Initialise the CausalGraph object."""
        self.time_series = time_series
        self.time_series_ensemble = time_series_ensemble
        self.standardize = standardize
        self.num_variables, self.periods = self._check_and_initialize_args(periods)
        self.seed = seed

        # outputs
        self.imbs_training = None
        self.weights_training = None
        self.weights_final = None
        self.imbs_final = None
        self.adj_matrix = None
        self.community_dictionary = None

        # graph refinement (direct vs indirect links beteween communities)
        self.weights_final_refine = None
        self.communities_and_lags_refine = None
        self.imbs_training_refine = None
        self.imbs_final_refine = None

    def _check_and_initialize_args(self, periods):
        """Check input arguments to constructor of CausalGraph object."""
        if self.time_series is not None:
            if self.time_series_ensemble is not None:
                warnings.warn(
                    "You passed both 'time_series' and 'time_series_ensemble'; the latter will be ignored.",
                    stacklevel=2,
                )
                self.time_series_ensemble = None
            assert (
                self.time_series.ndim == 2
            ), f"'time_series' has shape {self.time_series.shape}, while the expected shape is (N_times, D)."
            data = self.time_series
        else:
            assert (
                self.time_series_ensemble is not None
            ), "Provide either 'time_series' or 'time_series_ensemble' to initialize the CausalGraph class."
            assert self.time_series_ensemble.ndim == 3, (
                f"'time_series_ensemble' has shape {self.time_series_ensemble.shape}, while the expected "
                + "shape is (N_trajs, N_times, D)."
            )
            data = self.time_series_ensemble
        num_variables = data.shape[-1]
        if periods is not None:
            periods = np.ones(num_variables) * np.array(periods)

        # standard deviation of each variable along the time series (or over all trajectories and times)
        std = np.std(data.reshape(-1, num_variables), ddof=1, axis=0)
        if self.standardize is True:
            # not in place, so that the input array is not modified
            if self.time_series is not None:
                self.time_series = self.time_series / std
            else:
                self.time_series_ensemble = self.time_series_ensemble / std
        elif (std != 1).any():
            warnings.warn(
                f"The {num_variables} variables of the input time series are not standardized.",
                stacklevel=2,
            )
        return num_variables, periods

    def _select_times(
        self,
        num_samples,
        time_lags,
        embedding_dim_present,
        embedding_dim_future,
        embedding_time,
    ):
        """Select the times t=0 at which the samples in the present space are taken.

        The first selectable time is (max(embedding_dim_present, embedding_dim_future) - 1) * embedding_time,
        so that the times before t=0 needed by the time-delay embeddings are available.

        Args:
            num_samples (int): number of times selected along the time series. Ignored if the data are provided
                through 'time_series_ensemble', where the samples are the N_trajs trajectories.
            time_lags, embedding_dim_present, embedding_dim_future, embedding_time: see the method
                'optimize_present_to_future'.

        Returns:
            t0s (np.ndarray(int) or int): array of shape (num_samples,) containing the times selected along
                'time_series' or, for 'time_series_ensemble', the single time (along axis=1) at which all the
                trajectories are sampled.
        """
        t_first = (max(embedding_dim_present, embedding_dim_future) - 1) * embedding_time
        if self.time_series is not None:
            n_times = self.time_series.shape[0]
            assert num_samples <= n_times - max(time_lags) - t_first, (
                f"Cannot extract {num_samples} samples from a time series of length {n_times}, with maximum "
                + f"time lag {max(time_lags)} and {t_first} initial times reserved for the time-delay "
                + "embeddings. Choose a smaller value of num_samples."
            )
            return np.linspace(
                t_first, n_times - max(time_lags) - 1, num_samples, dtype=int
            )
        if num_samples is not None:
            warnings.warn(
                "Argument 'num_samples' will be ignored, as the samples are the trajectories in "
                + "'time_series_ensemble'. To suppress this warning, set 'num_samples' to None.",
                stacklevel=3,
            )
        n_times = self.time_series_ensemble.shape[1]
        assert t_first + max(time_lags) < n_times, (
            f"The trajectories in 'time_series_ensemble' have {n_times} times, while the maximum time lag "
            + f"({max(time_lags)}) and the time-delay embeddings require at least {t_first + max(time_lags) + 1}."
        )
        return t_first

    def _extract_samples(self, t0s, time_shifts, variables):
        """Extract the samples of some variables at the times t0s + time_shifts.

        Args:
            t0s (np.ndarray(int) or int): times t=0 returned by the method '_select_times'.
            time_shifts (np.ndarray(int)): array of shape (n_shifts,) containing the time shifts with respect
                to t=0 (e.g. tau, tau-1, ... for time-delay embeddings in the future).
            variables (list(int), np.ndarray(int)): indices of the extracted variables.

        Returns:
            samples (np.ndarray(float)): array of shape (num_samples, n_variables * n_shifts), with columns
                ordered as (variable_1, shift_1), (variable_1, shift_2), ..., (variable_2, shift_1), ...
        """
        if self.time_series is not None:
            samples = self.time_series[
                np.add.outer(t0s, time_shifts)
            ]  # has shape (num_samples, n_shifts, D)
        else:
            samples = self.time_series_ensemble[
                :, t0s + time_shifts
            ]  # has shape (N_trajs, n_shifts, D)
        samples = np.transpose(
            samples[:, :, variables], axes=[0, 2, 1]
        )  # convert to shape (num_samples, n_variables, n_shifts)
        return samples.reshape((samples.shape[0], -1))

    def optimize_present_to_future(  # noqa: C901
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
        track_full_loss=False,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k=1,
        lambda_factor=0.1,
        params_init=None,
        optimizer_name="adam",
        learning_rate=1e-2,
        learning_rate_decay=None,
        learning_rate_final=None,
        compute_imb_final=False,
        discard_close_ind=0,
    ):
        """Optimize the DII iteratively from the full space in the present to a target space in the future.

        Argument 'num_samples' is read only when data are provided to the CausalGraph object through the argument
        'time_series'; with 'time_series_ensemble', the samples are the N_trajs trajectories.

        Args:
            num_samples (int): number of samples harvested from the full time series, interpreted as
                independent initial conditions of the same dynamical process. Ignored (set it to None) if the
                data are provided through 'time_series_ensemble'.
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
            track_full_loss (bool): whether to compute the training DII on the full data set at each training epoch,
                if minibatches are used (batches_per_epoch > 1). This can be computationally demanding for large
                data sets, but helps monitoring the convergence of the DII, as its calculation on small minibatches
                may be affected by large fluctuations. Default is False.
            l1_strength (float): strength of the L1 regularization (LASSO) term, currently supported only with
                optimizer_name='sgd' (if 'adam' is set, the optimizer is changed to 'sgd' with a warning).
                Default is 0.
            point_adapt_lambda (bool): whether to use a global smoothing parameter lambda for the c_ij coefficients
                in the DII (if False), or a different parameter for each point (if True). Default is False.
            k (int): distance rank of neighbors used to set lambda. Ranks are defined starting from 1. If
                batches_per_epoch > 1, neighbors are recomputed within each mini-batch. Default is 1.
            lambda_factor (float): factor defining the scale of lambda. Default is 0.1.
            params_init (np.array(float), jnp.array(float)): array of shape (n_features_A,) containing the initial
                values of the scaling weights to be optimized. If None, the initial weights are all equal, with unit
                norm: [1, 1, ..., 1] / sqrt(n_features_A). The norm of the weights is kept fixed during the training.
            optimizer_name (str): name of the optimizer, calling the Optax library. Possible choices are 'adam'
                (default) and 'sgd'. If l1_strength is not 0, only 'sgd' is supported. See
                https://optax.readthedocs.io/en/latest/api/optimizers.html for additional details.
            learning_rate (float): value of the learning rate. Default is 1e-2.
            learning_rate_decay (str): schedule to damp the learning rate to zero (or to learning_rate_final, if not
                None) starting from the value provided with the attribute learning_rate. The available schedules are:
                cosine decay ("cos"), or constant learning rate (None). Default is None (constant learning rate).
            learning_rate_final (float): final value of the learning rate when the "cos" decay schedule is applied.
                Default is None, for which the learning rate is damped to zero. If learning_rate_decay=None, this
                argument is ignored.
            compute_imb_final (bool): whether to compute the final DII over the full data set (see the method
                'return_final_dii' of the DiffImbalance class). Default is False.
            discard_close_ind (int): given any point i, defines the "close" points (following the order of the
                samples: the times selected along 'time_series', or the trajectories along axis=0 of
                'time_series_ensemble') that are known to be significantly correlated with i.
                The pairs (i, j) with |i - j| <= discard_close_ind are discarded both during the training and when
                computing the final DII (see the argument 'discard_close_ind' of the DiffImbalance class). Default is
                0, for which no distances between points close in the time are discarded.

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
        """
        t0s = self._select_times(
            num_samples,
            time_lags,
            embedding_dim_present,
            embedding_dim_future,
            embedding_time,
        )
        # present space: all variables at times t=0, -1, ... (time-delay embeddings)
        coords_present = self._extract_samples(
            t0s,
            -embedding_time * np.arange(embedding_dim_present),
            np.arange(self.num_variables),
        )

        if isinstance(target_variables, str) and target_variables == "all":
            target_variables = np.arange(self.num_variables)

        # initialize output variables
        imbs_training = np.zeros(
            (len(target_variables), len(time_lags), num_epochs + 1)
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
                        num_epochs + 1,
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
                        num_epochs + 1,
                        self.num_variables,
                        embedding_dim_present,
                    )
                )
        imbs_final = None
        if compute_imb_final:
            imbs_final = np.zeros((len(target_variables), len(time_lags)))

        # loop over target variables and time lags
        for i_var, target_var in enumerate(target_variables):
            for j_tau, tau in enumerate(time_lags):
                # future space: target variable at times t=tau, tau-1, ... (time-delay embeddings)
                coords_future = self._extract_samples(
                    t0s,
                    tau - embedding_time * np.arange(embedding_dim_future),
                    [target_var],
                )

                dii = DiffImbalance(
                    data_A=coords_present,
                    data_B=coords_future,
                    periods_A=(  # columns ordered by variable, then time shift
                        None
                        if self.periods is None
                        else np.repeat(self.periods, embedding_dim_present)
                    ),
                    periods_B=(
                        None if self.periods is None else self.periods[target_var]
                    ),
                    seed=self.seed,
                    num_epochs=num_epochs,
                    batches_per_epoch=batches_per_epoch,
                    track_full_loss=track_full_loss,
                    l1_strength=l1_strength,
                    point_adapt_lambda=point_adapt_lambda,
                    k=k,
                    lambda_factor=lambda_factor,
                    params_init=params_init,
                    optimizer_name=optimizer_name,
                    learning_rate=learning_rate,
                    learning_rate_decay=learning_rate_decay,
                    learning_rate_final=learning_rate_final,
                    discard_close_ind=discard_close_ind,
                )
                weights_temp, imbs_training[i_var, j_tau] = dii.train(
                    bar_label=f"target_var={target_var}, tau={tau}"
                )
                # weights enter the distances squared, so only their absolute value is meaningful
                weights_temp = np.abs(weights_temp)

                # compute final DII
                if compute_imb_final:
                    imbs_final[i_var, j_tau] = dii.return_final_dii()

                # save weights
                if embedding_dim_present == 1:
                    weights_final[i_var, j_tau] = weights_temp[-1]
                    if save_weights is True:
                        weights_training[i_var, j_tau] = weights_temp.reshape(
                            (num_epochs + 1, self.num_variables)
                        )
                elif embedding_dim_present > 1:
                    weights_final[i_var, j_tau] = weights_temp[-1].reshape(
                        (self.num_variables, embedding_dim_present)
                    )
                    if save_weights is True:
                        weights_training[i_var, j_tau] = weights_temp.reshape(
                            (num_epochs + 1, self.num_variables, embedding_dim_present)
                        )

        self.weights_final = weights_final
        self.imbs_training = imbs_training
        if save_weights:
            self.weights_training = weights_training
        self.imbs_final = imbs_final
        return weights_final, imbs_training, imbs_final

    def compute_adj_matrix(self, weights, threshold):
        """Compute the adjacency matrix from the optimal weights returned by optimize_present_to_future.

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
            "To call this method, provide the weights obtained with the method optimize_present_to_future, "
            + "with the option target_variables='all'"
        )
        assert len(weights.shape) == 3 or len(weights.shape) == 4, (
            "The array of weight must have shape (D,n_time_lags,D), or (D,n_time_lags,D,embedding_dim_present). "
            + "If you are testing a single time lag, reshape this input as weights[:,np.newaxis,:]."
        )
        assert weights.shape[0] == weights.shape[2], (
            "The array of weight must have shape (D,n_time_lags,D), or (D,n_time_lags,D,embedding_dim_present), "
            + "where D is the number of variables."
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
        """Find ancestors of each node in the directed graph described by the input adjacency matrix.

        Args:
            adj_matrix (np.ndarray(float)): binary matrix of shape (D,D) defining the links of a directed
                graph.

        Returns:
            auto_sets (list): list of lists, such that auto_sets[i] is a list containing the indices of all
                ancestors of node i in the graph
        """
        G = nx.DiGraph(adj_matrix)
        auto_sets = []
        for var in range(adj_matrix.shape[0]):
            auto_sets.append(sorted(nx.ancestors(G, var) | {var}))
        return auto_sets

    def find_communities(self, adj_matrix):
        """Find dynamical communities, i.e. groups of variables defining single nodes in the community causal graph.

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

    def community_graph_visualization(  # noqa: C901
        self,
        community_dictionary,
        adj_matrix,
        type="community",
        savefig_name=None,
        variable_names=None,
        **kwargs,
    ):
        """Show a visual representation of the dynamical communities on a graph.

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
            savefig_name (str): path at which the picture of the final graph is saved, in the format given by
                the file extension. If None (default), the figure is not saved.
            variable_names (np.array(str)): array of shape (D,) containing the names of the D variables.
                Used only if type="community", to show the variable names in the printout of each community.
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
            ), "Only one community is present. Try plotting with a standard function of networkx."

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
            alphabet_string = list(
                itertools.islice(symbol_generator(), adj_matrix.shape[0])
            )
            community_names = {
                tuple(community): alphabet_string[i]
                for i, community in enumerate(values)
            }

            # convert communities into names and add them to graph as nodes
            print("Conversion labels - communities:")
            for community, key in zip(community_names, keys):
                community_name = community_names[tuple(community)]
                G.add_node(str(community_name))
                if variable_names is None:
                    print(
                        f"Community {community_name} ({len(community)} variables, level {key[1]}): {community}"
                    )
                else:
                    print(
                        f"Community {community_name} ({len(community)} variables, "
                        f"level {key[1]}): {variable_names[list(community)]}"
                    )

            # dictionary with keys: (order_idx) and values: list of communities at that order (list of list)
            communities_orders = {key[1]: [] for key in keys}
            for community_idx, order_idx in keys:
                communities_orders[order_idx].append(
                    community_dictionary[community_idx, order_idx]
                )

            # draw edges
            for order_idx, communities in communities_orders.items():
                if order_idx == 0:
                    continue
                # for each putative effect community at order >=1...
                for community_effect in communities:
                    community_name_effect = community_names[tuple(community_effect)]
                    # ...loop over all putative causal communities at previous orders
                    for previous_order in range(0, order_idx):
                        for community_cause in communities_orders[previous_order]:
                            community_name_cause = community_names[
                                tuple(community_cause)
                            ]
                            # ...and draw an edge if at least a link is found
                            if adj_matrix[
                                np.ix_(community_cause, community_effect)
                            ].any():
                                G.add_edges_from(
                                    [
                                        (
                                            str(community_name_cause),
                                            str(community_name_effect),
                                        )
                                    ]
                                )

            # delete edges in presence of indirect paths
            G = nx.transitive_reduction(G)

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

    def find_direct_links_communities(  # noqa: C901
        self,
        adj_matrix,
        community_graph,
        community_dictionary,
        num_samples,
        time_lags,
        embedding_dim_present=1,
        embedding_dim_future=1,
        embedding_time=1,
        num_epochs=200,
        batches_per_epoch=1,
        track_full_loss=False,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k=1,
        lambda_factor=0.1,
        optimizer_name="adam",
        learning_rate=1e-2,
        learning_rate_decay=None,
        learning_rate_final=None,
        compute_imb_final=False,
        discard_close_ind=0,
    ):
        """Implement the refinement step to distinguish direct and indirect links between nonconsecutive communities.

        Whenever a pattern A->C->B is found in the community causal graph, the loss
            DII(w * [A(0), B(tau-1), C(tau-1),..., B(tau-E), C(tau-E)] -> B(tau))
        is optimized for all input time lags 'tau'. All variables within the same community are scaled
        by the same weight. The number of previous time steps E included in the optimization is given
        by the argument 'embedding_dim_present'.

        Argument 'num_samples' is read only when data are provided to the CausalGraph object through the argument
        'time_series'; with 'time_series_ensemble', the samples are the N_trajs trajectories.

        Args:
            adj_matrix (np.ndarray(float)): binary matrix of shape (D,D) defining the links of a directed
                graph with D nodes.
            community_graph (networkx.DiGraph): output of method 'community_graph_visualization', with option
                type="community".
            community_dictionary (dict): dictionary with pairs (comm_id, level) as keys, and lists containing
                the indices of the variables in each community as values.
            num_samples (int): number of samples harvested from the full time series, interpreted as
                independent initial conditions of the same dynamical process. Ignored (set it to None) if the
                data are provided through 'time_series_ensemble'.
            time_lags (list(int), np.ndarray(int)): tested time lags between 'present' and 'future'.
            embedding_dim_present (int): dimension of the time-delay embedding vectors built in the present
                space (t=0, t=-1, ...). Default is 1, which means the time-delay embeddings are not employed.
            embedding_dim_future (int): dimension of the time-delay embedding vectors built in the space of
                the target variable (t=tau, t=tau-1, ...). Default is 1.
            embedding_time (int): lag between consecutive samples in the time-delay embedding vectors of each
                variable.  Default is 1.
            num_epochs (int): number of training epochs. Default is 200.
            batches_per_epoch (int): number of minibatches; must be a divisor of n_points. Each weight update is
                carried out by computing the DII gradient over n_points / batches_per_epoch points. Default is 1,
                which means that the gradient is computed over all the available points (batch GD).
            track_full_loss (bool): whether to compute the training DII on the full data set at each training epoch,
                if minibatches are used (batches_per_epoch > 1). This can be computationally demanding for large
                data sets, but helps monitoring the convergence of the DII, as its calculation on small minibatches
                may be affected by large fluctuations. Default is False.
            l1_strength (float): strength of the L1 regularization (LASSO) term, currently supported only with
                optimizer_name='sgd' (if 'adam' is set, the optimizer is changed to 'sgd' with a warning).
                Default is 0.
            point_adapt_lambda (bool): whether to use a global smoothing parameter lambda for the c_ij coefficients
                in the DII (if False), or a different parameter for each point (if True). Default is False.
            k (int): distance rank of neighbors used to set lambda. Ranks are defined starting from 1. If
                batches_per_epoch > 1, neighbors are recomputed within each mini-batch. Default is 1.
            lambda_factor (float): factor defining the scale of lambda. Default is 0.1.
            optimizer_name (str): name of the optimizer, calling the Optax library. Possible choices are 'adam'
                (default) and 'sgd'. If l1_strength is not 0, only 'sgd' is supported. See
                https://optax.readthedocs.io/en/latest/api/optimizers.html for additional details.
            learning_rate (float): value of the learning rate. Default is 1e-2.
            learning_rate_decay (str): schedule to damp the learning rate to zero (or to learning_rate_final, if not
                None) starting from the value provided with the attribute learning_rate. The available schedules are:
                cosine decay ("cos"), or constant learning rate (None). Default is None (constant learning rate).
            learning_rate_final (float): final value of the learning rate when the "cos" decay schedule is applied.
                Default is None, for which the learning rate is damped to zero. If learning_rate_decay=None, this
                argument is ignored.
            compute_imb_final (bool): whether to compute the final DII over the full data set (see the method
                'return_final_dii' of the DiffImbalance class). Default is False.
            discard_close_ind (int): given any point i, defines the "close" points (following the order of the
                samples: the times selected along 'time_series', or the trajectories along axis=0 of
                'time_series_ensemble') that are known to be significantly correlated with i.
                The pairs (i, j) with |i - j| <= discard_close_ind are discarded both during the training and when
                computing the final DII (see the argument 'discard_close_ind' of the DiffImbalance class). Default is
                0, for which no distances between points close in the time are discarded.

        Returns:
            weights_final (dict): dictionary containing the final optimization weights for each pair
                of communities linked through indirect paths in the community causal graph. The keys
                are tuples (community_name_cause, community_name_effect), while the values are
                np.arrays of shape (n_time_lags, n_weights), where n_weights is equal to 1 + E + E*M
                (1 weight for the cause community at t=0, E weights for the effect community at
                t=tau-1, tau-1-embedding_time, ..., and E weights for each of the M mediator communities,
                ordered by time and then by community name), and E=embedding_dim_present. Each weight is
                shared by all the variables of a community at a given time.
            communities_and_lags (dict): dictionary containing as keys the tuples
                (community_name_cause, community_name_effect), and as values a list of two arrays of
                shape (n_weights,), containing the community name and the time of each weight.
            imbs_training (dict): dictionary containing as keys the tuples
                (community_name_cause, community_name_effect), and as values the DIIs during the
                trainings.
            imbs_final (dict): dictionary containing as keys the tuples
                (community_name_cause, community_name_effect), and as values the final DIIs.
        """

        def find_mediators(graph, node_start, node_end):
            all_paths = list(
                nx.all_simple_paths(graph, source=node_start, target=node_end)
            )
            if not all_paths:
                return []

            # Get intermediates (excluding A and B) for each path
            intermediates_per_path = [set(path[1:-1]) for path in all_paths]

            # Find union across all path intermediates (sorted, for a reproducible order of the weights)
            mediators = sorted(set.union(*intermediates_per_path))
            return mediators

        keys = list(community_dictionary.keys())
        values = list(community_dictionary.values())
        alphabet_string = list(
            itertools.islice(symbol_generator(), adj_matrix.shape[0])
        )
        community_names = {
            tuple(community): alphabet_string[i] for i, community in enumerate(values)
        }
        from_names_to_communities = {
            community_names[key]: key for key in community_names.keys()
        }

        # dictionary with keys: (order_idx) and values: list of communities at that order (list of list)
        communities_orders = {key[1]: [] for key in keys}
        for community_idx, order_idx in keys:
            communities_orders[order_idx].append(
                community_dictionary[community_idx, order_idx]
            )

        # print names of communities
        print("Conversion labels - communities:")
        for community, key in zip(community_names, keys):
            community_name = community_names[tuple(community)]
            print(
                f"- Community {community_name} ({len(community)} variables, level {key[1]}): {community}"
            )

        t0s = self._select_times(
            num_samples,
            time_lags,
            embedding_dim_present,
            embedding_dim_future,
            embedding_time,
        )

        # initialize output variables
        imbs_training = {}
        weights_final = {}
        imbs_final = {}
        communities_and_lags = {}

        # identify all pairs of indirectly linked communities, and mediator communities #############
        for order_idx, communities in communities_orders.items():
            if order_idx < 2:
                continue
            # for each putative effect community at order >=2...
            for community_effect in communities:
                community_name_effect = community_names[tuple(community_effect)]

                # ...find all its ancestor communities...
                effect_ancestors_names = set(
                    nx.ancestors(community_graph, community_name_effect)
                )

                # ...and take each ancestor (not a parent!) community as putative cause...
                for community_name_cause in list(
                    effect_ancestors_names.difference(
                        list(community_graph.predecessors(community_name_effect))
                    )
                ):
                    community_cause = list(
                        from_names_to_communities[community_name_cause]
                    )

                    # skip the test if there are no links in the adjacency matrix
                    if (
                        adj_matrix is not None
                        and (
                            adj_matrix[community_cause, :][:, community_effect] == 0
                        ).all()
                    ):
                        print(
                            f"Communities {community_name_cause}->{community_name_effect}: "
                            "not linked according to adjacency matrix, test skipped."
                        )
                        continue

                    # find mediating communities between cause and effect
                    mediator_names = find_mediators(
                        community_graph, community_name_cause, community_name_effect
                    )

                    # groups of variables sharing the same weight in space A, in the order of the output weights:
                    # the cause community at t=0, the effect community at t=tau-1, tau-1-embedding_time, ...,
                    # and all the mediator communities at t=tau-1, then at t=tau-1-embedding_time, ...
                    # Each group is a pair (community name, lag), with t=tau-lag (lag=None for t=0).
                    lags_cond = 1 + embedding_time * np.arange(embedding_dim_present)
                    groups_A = (
                        [(community_name_cause, None)]
                        + [(community_name_effect, lag) for lag in lags_cond]
                        + [(name, lag) for lag in lags_cond for name in mediator_names]
                    )
                    variables_A = [
                        list(from_names_to_communities[name]) for name, _ in groups_A
                    ]
                    params_groups = [len(variables) for variables in variables_A]

                    # initialize output variables
                    imbs_training[community_name_cause, community_name_effect] = (
                        np.zeros((len(time_lags), num_epochs + 1))
                    )
                    weights_final[community_name_cause, community_name_effect] = (
                        np.zeros((len(time_lags), len(groups_A)))
                    )
                    communities_and_lags[
                        community_name_cause, community_name_effect
                    ] = [
                        np.array([name for name, _ in groups_A]),
                        np.array(
                            [
                                "t=0" if lag is None else f"t=tau-{lag}"
                                for _, lag in groups_A
                            ]
                        ),
                    ]
                    if compute_imb_final:
                        imbs_final[community_name_cause, community_name_effect] = (
                            np.zeros(len(time_lags))
                        )

                    # LOOP OVER TAU #############
                    for j_tau, tau in enumerate(time_lags):
                        # space A: cause(t=0), effect(t=tau-1), ..., mediators(t=tau-1), ... (one block per group)
                        coords_A = np.concatenate(
                            [
                                self._extract_samples(
                                    t0s,
                                    np.array([0 if lag is None else tau - lag]),
                                    variables,
                                )
                                for (_, lag), variables in zip(groups_A, variables_A)
                            ],
                            axis=1,
                        )
                        # space B: effect at t=tau, tau-embedding_time, ... (time-delay embeddings)
                        coords_future = self._extract_samples(
                            t0s,
                            tau - embedding_time * np.arange(embedding_dim_future),
                            community_effect,
                        )

                        dii = DiffImbalance(
                            data_A=coords_A,
                            data_B=coords_future,
                            periods_A=(
                                None
                                if self.periods is None
                                else self.periods[np.concatenate(variables_A)]
                            ),
                            periods_B=(
                                None
                                if self.periods is None
                                else np.repeat(
                                    self.periods[community_effect],
                                    embedding_dim_future,
                                )
                            ),
                            seed=self.seed,
                            num_epochs=num_epochs,
                            batches_per_epoch=batches_per_epoch,
                            track_full_loss=track_full_loss,
                            l1_strength=l1_strength,
                            point_adapt_lambda=point_adapt_lambda,
                            k=k,
                            lambda_factor=lambda_factor,
                            params_init=None,
                            params_groups=params_groups,
                            optimizer_name=optimizer_name,
                            learning_rate=learning_rate,
                            learning_rate_decay=learning_rate_decay,
                            learning_rate_final=learning_rate_final,
                            discard_close_ind=discard_close_ind,
                        )
                        (
                            weights_temp,
                            imbs_training[community_name_cause, community_name_effect][
                                j_tau
                            ],
                        ) = dii.train(
                            bar_label=f"Communities {community_name_cause}->{community_name_effect}, tau={tau}"
                        )
                        # weights enter the distances squared, so only their absolute value is meaningful
                        weights_temp = np.abs(weights_temp)

                        # compute final DII
                        if compute_imb_final:
                            imbs_final[community_name_cause, community_name_effect][
                                j_tau
                            ] = dii.return_final_dii()

                        # save weights
                        weights_final[community_name_cause, community_name_effect][
                            j_tau
                        ] = weights_temp[-1]

        self.weights_final_refine = weights_final
        self.communities_and_lags_refine = communities_and_lags
        self.imbs_training_refine = imbs_training
        self.imbs_final_refine = imbs_final
        return (
            weights_final,
            communities_and_lags,
            imbs_training,
            imbs_final,
        )

    def community_graph_refinement(
        self,
        community_graph,
        community_dictionary,
        weights_refine,
        communities_and_lags,
        variable_names,
        threshold,
        savefig_name=None,
        **kwargs,
    ):
        """Show a visual representation of the dynamical communities on a graph, after the refinement step.

        This function makes use of the library networkx (https://networkx.org/documentation/stable/index.html)

        Args:
            community_graph (networkx.DiGraph): output of method 'community_graph_visualization', with option
                type="community".
            community_dictionary (dict): dictionary with pairs (comm_id, level) as keys and lists containing
                the indices of the variables in each dynamical community as values.
            weights_refine (dict): output weights of method 'find_direct_links_communities'.
            communities_and_lags (dict): output communities and lags of method 'find_direct_links_communities'.
            variable_names (np.array(str)): array of shape (D,) containing the names of the D variables.
            threshold (float): weight threshold above which a direct link between two communities is drawn.
            savefig_name (str): path at which the picture of the final graph is saved, in the format given by
                the file extension. If None (default), the figure is not saved.
            **kwargs: customizable arguments used by the networkx library. The possible arguments are:
                'node_color', 'node_size', 'width', 'arrowstyle', 'arrowsize'.

        Returns:
            G (nx.diGraph object): refined community causal graph.
        """
        assert (
            community_dictionary is not None
        ), "Provide as intput the community dictionary computed with the method find_communities"

        # construct graph
        G = community_graph.copy()

        # read community dictionary and extract conversion
        keys = list(community_dictionary.keys())
        values = list(community_dictionary.values())
        alphabet_string = list(itertools.islice(symbol_generator(), len(keys)))
        community_names = {
            tuple(community): alphabet_string[i] for i, community in enumerate(values)
        }

        # print conversion labels - communities
        print("Conversion labels - communities:")
        for community, key in zip(community_names, keys):
            community_name = community_names[tuple(community)]
            if variable_names is None:
                print(
                    f"Community {community_name} ({len(community)} variables, level {key[1]}): {community}"
                )
            else:
                print(
                    f"Community {community_name} ({len(community)} variables, "
                    f"level {key[1]}): {variable_names[list(community)]}"
                )

        # extract pairs of communities tested for direct vs indirect links
        pairs_cause_effect = list(weights_refine.keys())

        # loop over such pairs and connect them when at least one weight of causal community > threshold
        for community_name_cause, community_name_effect in pairs_cause_effect:
            mask_variables_cause = (
                communities_and_lags[community_name_cause, community_name_effect][0]
                == community_name_cause
            )

            max_weights_refine = np.max(
                weights_refine[community_name_cause, community_name_effect],
                axis=0,  # axis of lag tau
            )[mask_variables_cause]
            if (max_weights_refine > threshold).any():
                G.add_edges_from(
                    [
                        (
                            str(community_name_cause),
                            str(community_name_effect),
                        )
                    ]
                )

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
