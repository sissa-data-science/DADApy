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

"""
The *feature_weighting* module contains the *FeatureWeighting* class.

This class uses Differentiable Information Imbalance
"""

import multiprocessing
import time
import warnings
from functools import wraps
from typing import Type, Union

import numpy as np
from scipy.linalg import norm

from dadapy._utils.differentiable_imbalance import (
    _extract_min_diis_lasso_optimization,
    _optimize_dii,
    _optimize_dii_static_zeros,
    _plot_min_lasso_results,
    _refine_lasso_optimization,
    _return_dii,
    _return_dii_gradient,
    _return_full_dist_matrix,
    _return_full_rank_matrix,
    _return_optimal_lambda_from_distances,
)
from dadapy.base import Base

cores = multiprocessing.cpu_count()


def check_maxk(func):
    # TODO: remove when this works with different maxk
    @wraps(func)
    def with_check(*args, **kwargs):
        feature_selector: type[FeatureWeighting] = args[0]
        if feature_selector._maxk_warning:
            if feature_selector.maxk != feature_selector.N - 1:
                warnings.warn(
                    f'{"maxk option not yet available for the FeatureWeighting class. "}'
                    + f"It will be set to the number of data-1 ({feature_selector.N}-1).",
                    stacklevel=2,
                )
            feature_selector._maxk_warning = False
        return func(*args, **kwargs)

    return with_check


class FeatureWeighting(Base):
    def __init__(
        self,
        coordinates=None,
        distances=None,
        maxk=None,
        period=None,
        verbose=False,
        n_jobs=cores,
    ):
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            period=period,
            verbose=verbose,
            n_jobs=n_jobs,
        )

        # This is quite useful for debugging
        self._cythond = True
        self.history = None
        self._full_distance_matrix = None

        # To show maxk warning only once
        self._maxk_warning = True

    @property
    def full_distance_matrix(self):
        if self._full_distance_matrix is None:
            self._full_distance_matrix = _return_full_dist_matrix(
                data=self.X,
                n_jobs=self.n_jobs,
                period=self._parse_own_period(),
                cythond=self._cythond,
            )
        return self._full_distance_matrix

    @full_distance_matrix.setter
    def full_distance_matrix(self, distance_matrix: np.ndarray):
        if (
            (len(distance_matrix.shape) != 2)
            or (distance_matrix.shape[0] != self.N)
            or (distance_matrix.shape[1] != self.N)
        ):
            raise ValueError(
                f"Input matrix for full distance matrix not properly shaped. \
                Should be {self.N}x{self.N} but is {distance_matrix.shape[0]}x{distance_matrix.shape[1]}."
            )
        self._full_distance_matrix = distance_matrix

    @staticmethod
    def _parse_period_for_dii(in_period, in_dims):
        # TODO: remove when part of Base
        if in_period is None:
            return None

        if isinstance(in_period, np.ndarray) and in_period.shape == (in_dims,):
            period = in_period
        elif isinstance(in_period, (int, float)):
            period = np.full((in_dims), fill_value=in_period, dtype=float)
        else:
            raise ValueError(
                f"'period' must be either a float scalar or a numpy array of floats of shape ({in_dims},)"
            )
        return period

    def _parse_own_period(self):
        return self._parse_period_for_dii(self.period, self.dims)

    def _parse_initial_weights(self, initial_weights: Union[np.ndarray, int, float]):
        if not (
            isinstance(initial_weights, np.ndarray)
            or isinstance(initial_weights, int)
            or isinstance(initial_weights, float)
            or initial_weights is None
        ):
            raise ValueError(
                f"'initial_weights' must be either None,"
                f" float scalar or a numpy array of floats of shape ({self.dims},)"
            )
        if initial_weights is not None:
            if isinstance(initial_weights, np.ndarray):
                if initial_weights.shape == (self.dims,):
                    initial_weights = initial_weights
                else:
                    raise ValueError(
                        f"'initial_weights' must be either None,"
                        f" float scalar or a numpy array of floats of shape ({self.dims},)"
                    )
            elif isinstance(initial_weights, (int, float)):
                initial_weights = np.full(
                    (self.dims), fill_value=initial_weights, dtype=float
                )
        else:
            initial_weights = 1 / np.std(self.X, axis=0)

        return initial_weights

    @check_maxk
    def return_optimal_lambda(self, fraction: float = 1.0):
        """Computes the optimal softmax scaling parameter lambda for the DII optimization.
        This parameter represents a reasonable scale of distances of the data points in the input data set.
        Args:
            fraction (float): Zoom in or out from the optimal distance scale.
                Default: 1.0. Suggested to keep it at default.
                Values > 1. show a bigger scale (in the optimization, this means include more neigbors),
                values < 1 show a smaller scale (in the optimization, this means include less neighbors in the softmax).
                Values < 1. include on average less neighbors, and very small values only the first neighbor
        """
        return _return_optimal_lambda_from_distances(
            self.full_distance_matrix, fraction
        )

    @check_maxk
    def return_optimal_learning_rate(
        self,
        target_data: Type[Base],
        n_epochs: int = 50,
        n_samples: int = 200,
        initial_weights: Union[np.ndarray, int, float] = None,
        lambd: float = None,
        decaying_lr: bool = True,
        trial_learning_rates: np.ndarray = None,
    ):
        """Find the optimal learning rate for the optimization of the DII by testing several on a reduced set
        Args:
            target_data: FeatureWeighting object, containing the
                groundtruth data (D_groundtruth x N array, period (optional)) to be compared to.
            n_epochs (int): number of epochs in each optimization cycle
            n_samples (int): Number of samples to use for the learning rate screening. Default = 300.
            initial_weights (np.ndarray or list): D(input) initial weights for the input features. No zeros allowed here
            lambd (float): softmax scaling. If None (preferred),
                this chosen automatically with compute_optimial_lambda
            decaying_lr (bool): default: True.
                Apply decaying learning rate = l_rate * 2**(-i_epoch/10)
                - every 10 epochs the learning rate will be halfed
            trial_learning_rates (np.ndarray or list or None): learning rates to try.
                If None are given, a sensible set of learning rates is tested.
        Returns:
            opt_l_rate (float): Learning rate,
                which leads to optimal unregularized (no l1-penalty) result in the specified number of epochs.

        History entries added to FeatureWeighting object:
            trial_learning_rates: np.ndarray. learning rates which were tested to find optimal one.
            dii_per_epoch_per_lr: np.ndarray, shape (len(trial_learning_rates), n_epochs+1).
                DII for each trial learning rate at each epoch.
            weights_per_epoch_per_lr: np.ndarray, shape (len(trial_learning_rates), n_epochs+1, D).
                Weights for each trial learning rate and at each epoch.
        These history entries can be accessed as follows: objectname.history['entry_name']
        """
        in_data = self.X.copy()
        groundtruth = target_data.X.copy()

        if n_samples <= len(in_data):
            in_data = in_data[-n_samples:]
            groundtruth = groundtruth[-n_samples:]

        initial_weights = self._parse_initial_weights(initial_weights)
        period = self._parse_own_period()
        groundtruthperiod = self._parse_period_for_dii(
            target_data.period, target_data.dims
        )

        if trial_learning_rates is None:
            # these learning rates seem to work well for most data
            lrates = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 200.0])
        else:
            lrates = trial_learning_rates

        weights_per_epoch_per_lr = np.zeros(
            (len(lrates), n_epochs + 1, in_data.shape[1])
        )
        dii_per_epoch_per_lr = np.zeros((len(lrates), n_epochs + 1))

        # optmizations for different learning rates
        for i, lrate in enumerate(lrates):
            (
                weights_per_epoch_per_lr[i],
                dii_per_epoch_per_lr[i],
                _,
            ) = _optimize_dii(
                groundtruth_data=groundtruth,
                data=in_data,
                weights_0=initial_weights,
                lambd=lambd,
                n_epochs=n_epochs,
                l_rate=lrate,
                constrain=False,
                l1_penalty=0.0,
                decaying_lr=decaying_lr,
                period=period,
                groundtruthperiod=groundtruthperiod,
                n_jobs=self.n_jobs,
                cythond=self._cythond,
            )

        # find best imbalance
        opt_lrate_index = np.nanargmin(dii_per_epoch_per_lr[:, -1])
        opt_l_rate = lrates[opt_lrate_index]

        self.history = {
            "dii_per_epoch_per_lr": dii_per_epoch_per_lr,
            "weights_per_epoch_per_lr": weights_per_epoch_per_lr,
            "trial_learning_rates": lrates,
        }
        return opt_l_rate

    @check_maxk
    def return_dii(self, target_data: Type[Base], lambd: float = None):
        """Computes the DII between two FeatureWeighting objects based
            on distances of input data and rank information of groundtruth data.

        Args:
            target_data: FeatureWeighting object,
                containing the groundtruth data (D_groundtruth x N array, period (optional)) to be compared to.
            lambd (float, optional): The regularization parameter. Default: 0.1.
                The higher this value, the more nearest neighbors are included.
                Can be calculated automatically with 'return_optimal_lambda'.
                This sets lambda to a distance smaller than the average distance
                in the data set but bigger than the minimal distance

        Returns:
            dii (float): The computed DII value. Depends on the softmax scale lambda.

        Raises:
            None.
        """
        # only accepting target data of Base (or children) is slow if base automatically calculates distances.
        # either add lazyBase or find other way to implement things like period and metric of Base.
        if lambd is None:
            lambd = self.return_optimal_lambda()

        distances_i = self.full_distance_matrix
        rank_matrix_j = _return_full_rank_matrix(
            target_data.X,
            period=self._parse_period_for_dii(target_data.period, target_data.dims),
            n_jobs=self.n_jobs,
        )

        return _return_dii(
            dist_matrix_A=distances_i, rank_matrix_B=rank_matrix_j, lambd=lambd
        )

    @check_maxk
    def return_dii_gradient(
        self, target_data: Type[Base], weights: np.ndarray, lambd: float = None
    ):
        """Computes the gradient of the DII between two FeatureWeighting objects
            (input object and ground truth object (= target_data)) with respect to the weights of the input features.

        Args:
            target_data: FeatureWeighting object, containing the groundtruth data
                (D_groundtruth x N array, period (optional)) to be compared to.
            weights (np.ndarray): The array of weight values for the input values, where D is the dimension of data.
            lambd (float, optional): The regularization parameter. Default: 0.1.
                The higher this value, the more nearest neighbors are included.
                Can be calculated automatically with 'return_optimal_lambda'.
                This sets lambda to a distance smaller than the average distance
                in the data set but bigger than the minimal distance

        Returns:
            dii_weight_gradient (np.ndarray): The computed gradient of DII with respect to the weights.
                Depends on the softmax scale lambda.
        """
        if lambd is None:
            lambd = self.return_optimal_lambda()

        period = self._parse_own_period()
        if period is not None:
            period *= weights
        target_period = self._parse_period_for_dii(
            target_data.period, in_dims=target_data.dims
        )

        rescaled_distances_i = _return_full_dist_matrix(
            self.X * weights, period=period, n_jobs=self.n_jobs
        )
        rank_matrix_j = _return_full_rank_matrix(
            target_data.X, period=target_period, n_jobs=self.n_jobs
        )

        return _return_dii_gradient(
            rescaled_distances_i,
            self.X,
            rank_matrix_j,
            weights=self._parse_initial_weights(weights),
            lambd=lambd,
            period=period,
            n_jobs=self.n_jobs,
            cythond=self._cythond,
        )

    @check_maxk
    def return_weights_optimize_dii(
        self,
        target_data: Type[Base],
        n_epochs: int = 100,
        constrain: bool = False,
        initial_weights: Union[np.ndarray, int, float] = None,
        lambd: float = None,
        learning_rate: float = None,
        l1_penalty: float = 0.0,
        decaying_lr: bool = True,
    ):
        """Optimize the differentiable information imbalance using gradient descent
            of the DII between input data object A and groundtruth data object B.

        Args:
            target_data: FeatureWeighting object, containing the groundtruth data
                (D_groundtruth x N array, period (optional)) to be compared to.
            n_epochs: int, optional
                The number of epochs in the gradient descent optimization. If None, it is set to 100.
            constrain: bool
                Constrain the sum of the weights to sum up to the number of weights. Default: False
            initial_ weights : numpy.ndarray, shape (D,)
                The array of starting weight values for the input values, where D is the dimension of data.
                If none, it is initialized to 1/var for each variable
                This cannot be initialized to 0's.
                It can be initialized to all 1 or the inverse of the standard deviation
            lambd : float, optional
                The lambda scaling parameter of the softmax. If None, it is calculated automatically. Default is None.
            learning_rate: float, optional
                The learning rate of the gradient descent. If None, automatically estimated to be fast.
            l1_penalty: float, optional
                The l1-regularization strength, if sparcity is needed. Default: 0 (l1-regularization turned off).
            decaying_lr: bool
                Use exponentially decaying learning rate in gradient descent or not. Default: True.

        Returns:
            final_weights: np.ndarray, shape (D). Array of the optmized weights.

        History entries added to FeatureWeighting object:
            weights_per_epoch: np.ndarray, shape (n_epochs+1, D).
                List of lists of the weights during optimization.
            dii_per_epoch: np.ndarray, shape (n_epochs+1, ).
                List of the differentiable information imbalances during optimization.
            l1_term_per_epoch: np.ndarray, shape (n_epochs+1, ).
                List of the l1_penalty terms contributing to the the loss function during optimization.
        These history entries can be accessed as follows: objectname.history['entry_name']
        """
        # initiate the weights
        period = self._parse_own_period()
        initial_weights = self._parse_initial_weights(initial_weights)

        # find a suitable learning rate by chosing the best optimization
        if learning_rate is None:
            learning_rate = self.return_optimal_learning_rate(
                target_data=target_data,
                n_epochs=50,
                n_samples=200,
                initial_weights=initial_weights,
                lambd=lambd,
                decaying_lr=decaying_lr,
                trial_learning_rates=None,
            )

        weights_list, diis, l1_loss_terms = _optimize_dii(
            groundtruth_data=target_data.X,
            groundtruthperiod=self._parse_period_for_dii(
                target_data.period, target_data.dims
            ),
            data=self.X,
            period=period,
            weights_0=initial_weights,
            lambd=lambd,
            constrain=constrain,
            l1_penalty=l1_penalty,
            n_epochs=n_epochs,
            l_rate=learning_rate,
            decaying_lr=decaying_lr,
            n_jobs=self.n_jobs,
            cythond=self._cythond,
        )
        # TODO: include a function that gives at least a reasonable estimate for the l1 penalty when wanting x features
        self.history = {
            "weights_per_epoch": weights_list,
            "dii_per_epoch": diis,
            "l1_term_per_epoch": l1_loss_terms,
        }
        return weights_list[-1]

    @check_maxk
    def return_backward_greedy_dii_elimination(
        self,
        target_data: Type[Base],
        initial_weights: Union[np.ndarray, int, float] = None,
        lambd: float = None,
        n_epochs: int = 100,
        learning_rate: float = None,
        constrain: bool = False,
        decaying_lr: bool = True,
    ):
        """Do a stepwise backward elimination of feature weights, always eliminating the lowest weight;
            after each elimination the DII is optimized by gradient descent using the remaining features

        Args:
            target_data: FeatureWeighting object, containing the groundtruth data
                (D_groundtruth x N array, period (optional)) to be compared to.
            initial_weights (np.ndarray or list): D(input) initial weights for the input features. No zeros allowed here
            lambd (float): softmax scaling. If None (preferred) this chosen automatically with compute_optimal_lambda
            n_epochs (int): number of epochs in each optimization cycle
            learning_rate (float): learning rate.
                Has to be tuned, especially if constrain=True (otherwise optmization could fail)
            constrain (bool): if True, rescale the weights so the biggest weight = 1
            l1_penalty (float): l1 regularization strength
            decaying_lr (bool): default: True. Apply decaying learning rate = l_rate * 2**(-i_epoch/10)
                - every 10 epochs the learning rate will be halfed

        Returns:
            final_diis: np.ndarray, shape (D). Array of the optmized DII for each of the according weights.
            final_weights: np.ndarray, shape (D x D). Array of the optmized weights for each number of non-zero weights.

        History entries added to FeatureWeighting object:
            dii_per_epoch: np.ndarray, shape (D, n_epochs+1, D).
                Weights during optimisation for every epoch and every number of non-zero weights.
                For final weights: weights_list[:,-1,:]
            weights_per_epoch: np.ndarray, shape (D, n_epochs+1, ).
            DII during optimization for every epoch and number of non-zero weights.
            For final imbalances: diis_list[:,-1]
        These history entries can be accessed as follows: objectname.history['entry_name']
        """
        initial_weights = self._parse_initial_weights(initial_weights)

        # INFO: do not precompute optimal lambda here, otherwise it becomes a fixed value in the optimization
        # and the results are not optimal any more.
        if learning_rate is None:
            learning_rate = self.return_optimal_learning_rate(
                target_data=target_data,
                n_epochs=50,
                n_samples=200,
                initial_weights=initial_weights,
                lambd=lambd,
                decaying_lr=decaying_lr,
                trial_learning_rates=None,
            )

        weights_per_epoch = np.full((self.dims, n_epochs + 1, self.dims), np.nan)
        imbalances_per_epoch = np.full((self.dims, n_epochs + 1), np.nan)
        # for making a warm start already for the first optimization
        end_weights = self.return_weights_optimize_dii(
            target_data=target_data,
            n_epochs=n_epochs,
            initial_weights=initial_weights,
            lambd=lambd,
            learning_rate=learning_rate,
            decaying_lr=decaying_lr,
            l1_penalty=0.0,
        )
        nonzeros = norm(end_weights, 0)

        while nonzeros >= 1:
            start = time.time()
            gs, imbs = _optimize_dii_static_zeros(
                groundtruth_data=target_data.X,
                data=self.X,
                weights_0=end_weights,
                lambd=lambd,
                n_epochs=n_epochs,
                l_rate=learning_rate,
                constrain=constrain,
                decaying_lr=decaying_lr,
                period=self._parse_own_period(),
                groundtruthperiod=self._parse_period_for_dii(
                    target_data.period, target_data.dims
                ),
                n_jobs=self.n_jobs,
                cythond=self._cythond,
            )

            end = time.time()
            timing = end - start
            if self.verb:
                print(
                    f"number of nonzero weights: {int(nonzeros)}, execution time: {timing:.2f} s."
                )
            end_weights = gs[-1].copy()
            arr = end_weights.copy()
            arr[arr == 0] = np.nan
            if np.isnan(arr).all():
                weights_per_epoch[self.dims - int(nonzeros)] = gs
                imbalances_per_epoch[self.dims - int(nonzeros)] = imbs
                break
            minweight = np.nanargmin(arr)
            end_weights[minweight] = 0
            weights_per_epoch[self.dims - (int(nonzeros))] = gs
            imbalances_per_epoch[self.dims - (int(nonzeros))] = imbs
            nonzeros = norm(end_weights, 0)

        self.history = {
            "dii_per_epoch": imbalances_per_epoch,
            "weights_per_epoch": weights_per_epoch,
        }

        # to select a sensible set of features, plot the imbalances
        # and chose at which number of non-zero weights still enough information is retained
        return imbalances_per_epoch[:, -1], weights_per_epoch[:, -1, :]

    @check_maxk
    def return_lasso_optimization_dii_search(
        self,
        target_data: Type[Base],
        initial_weights: Union[np.ndarray, int, float] = None,
        lambd: float = None,
        n_epochs: int = 100,
        learning_rate: float = None,
        l1_penalties: Union[list, float] = None,
        constrain: bool = False,
        decaying_lr: bool = True,
        refine: bool = False,
        plotlasso: bool = True,
    ):
        """Search the number of resulting non-zero weights and the optimized DII for several l1-regularization strengths
        Args:
            target_data: FeatureWeighting object, containing the groundtruth data
                (D_groundtruth x N array, period (optional)) to be compared to.
            initial_weights (np.ndarray or list): D(input) initial weights for the input features.
                No zeros allowed. If None (default), the inverse standard deviation of the input features is used
            lambd (float or None): softmax scaling.
                If None (default), lambd is chosen automatically with compute_optimial_lambda.
            n_epochs (int): number of epochs in each optimization cycle. Default: 100.
            learning_rate (float or None): learning rate.
                If None (default) is tuned and chosen automatically.
                Has to be tuned if constrain=True (otherwise optmization could fail).
            constrain (bool): if True, rescale the weights so the biggest weight = 1. Default: False.
            l1_penalties (list or None): l1 regularization strengths to be tested.
                If None (default), a list of 10 sensible l1-penalties is tested,
                which are chosen depending on the learning rate.
            decaying_lr (bool): default: True. Apply decaying learning rate = l_rate * 2**(-i_epoch/10)
                - every 10 epochs the learning rate will be halfed.
            refine (bool): default: False. If True, the l1-penalties are added in between penalties
                where the number of non-zero weights changes by more than one.
                This is done to find the optimal l1-penalty for each number of non-zero weights.
                This option is not suitable for high-dimensional data with more than ~100 features,
                because the computational time scales with the number of dimensions.
            plotlasso (bool): default: True. If True, a plot is shown,
                with the optimal DII for each number of non-zero weights,
                colored by the l1-penalty used. This plot can be used to select select results with reasonably low DII.

        Returns:
            num_nonzero_features (np.ndarray): D-dimensional numbers of non-zero features.
                Returns nan if no solution was found for a certain number of non-zero weights.
                In the same order as the according l1-penalties used, final DIIs and final weights.
            l1_penalties_opt_per_nfeatures: (np.ndarray): D-dimensional.
                L1-regularization strengths for each num_nonzero_features,
                in the same order as the according final DIIs and final weights.
                If several l1-penalties led to the same number of non-zero weights,
                the solution with the lowest DII is selected.
                Returns nan if no solution was found for a certain number of non-zero weights.
            dii_opt_per_nfeatures: (np.ndarray): D-dimensional.
                Final DIIs for each num_nonzero_features,
                in the same order as the according l1-penalties used and final weights.
                Returns nan if no solution was found for a certain number of non-zero weights.
            weights_opt_per_nfeatures: (np.ndarray): D x D-dimensional.
                Final weights for each num_nonzero_features,
                in the same order as the according l1-penalties used and final DIIs used.
                Returns nan if no solution was found for a certain number of non-zero weights.


        History entries added to FeatureWeighting object:
            l1_penalties (np.ndarray): len(l1_penalties). The l1-regularization strengths tested
                (in the order of the returned weights, diis and l1_loss_contributions)
            weights_per_l1_per_epoch (np.ndarray): len(l1_penalties) x n_epochs x D.
                All weights for each optimization step for each number of l1-regularization.
                For final weights: weights_list[:,-1,:]
            dii_per_l1_per_epoch (np.ndarray): len(l1_penalties) x n_epochs.
                Imbalance for each optimization step for each number of l1-regularization strength.
                For final imbalances: diis_list[:,-1]
            l1_term_per_l1_per_epoch (np.ndarray): len(l1_penalties) x n_epochs.
                L1 loss contributions for each optimization step for each number of nonzero weights.
                For final l1_loss_contributions: l1_loss_contributions[:,-1]
        These history entries can be accessed as follows: objectname.history['entry_name']
        """
        # Initial l1 search
        initial_weights = self._parse_initial_weights(initial_weights)

        # INFO: do not precompute optimal lambda here, otherwise it becomes a fixed value in the optimization
        # and the results are not optimal any more.

        if learning_rate is None:
            learning_rate = self.return_optimal_learning_rate(
                target_data=target_data,
                n_epochs=50,
                n_samples=200,
                initial_weights=initial_weights,
                lambd=lambd,
                decaying_lr=decaying_lr,
                trial_learning_rates=None,
            )

        if l1_penalties is None:
            l1_penalties = [0] + list(
                np.logspace(
                    np.floor(np.log10((1 / learning_rate) / 1000)),
                    np.ceil(np.log10((1 / learning_rate) * 1.5)),
                    9,
                )
            )  # test l1's depending on the learning rate
        elif isinstance(l1_penalties, (int, float)):
            l1_penalties = [l1_penalties]
        elif isinstance(l1_penalties, np.ndarray):
            l1_penalties = list(l1_penalties)

        weights = np.zeros((len(l1_penalties), n_epochs + 1, self.dims))
        diis = np.zeros((len(l1_penalties), n_epochs + 1))
        l1_loss_contributions = np.zeros((len(l1_penalties), n_epochs + 1))

        if self.verb:
            print(len(l1_penalties), "l1-penalties to test:")

        for i in range(len(l1_penalties)):
            start = time.time()

            weights[i], diis[i], l1_loss_contributions[i] = _optimize_dii(
                groundtruth_data=target_data.X,
                data=self.X,
                weights_0=initial_weights,
                lambd=lambd,
                n_epochs=n_epochs,
                l_rate=learning_rate,
                constrain=constrain,
                l1_penalty=l1_penalties[i],
                decaying_lr=decaying_lr,
                period=self._parse_own_period(),
                groundtruthperiod=self._parse_period_for_dii(
                    target_data.period, target_data.dims
                ),
                n_jobs=self.n_jobs,
                cythond=self._cythond,
            )

            end = time.time()
            if self.verb:
                print(
                    f"optimization with l1-penalty {i+1} of strength "
                    + f"{l1_penalties[i]:.4g} took: {end - start:.2f} s.",
                )

        # Refine l1 search
        if refine:
            (
                weights_list,
                dii_list,
                lassoterm_list,
                penalties,
            ) = _refine_lasso_optimization(
                weights,
                diis,
                l1_loss_contributions,
                l1_penalties,
                groundtruth_data=target_data.X,
                data=self.X,
                weights_0=initial_weights,
                lambd=lambd,
                n_epochs=n_epochs,
                l_rate=learning_rate,
                constrain=constrain,
                decaying_lr=decaying_lr,
                period=self._parse_own_period(),
                groundtruthperiod=self._parse_period_for_dii(
                    target_data.period, target_data.dims
                ),
                n_jobs=self.n_jobs,
                cythond=self._cythond,
                verbose=self.verb,
            )
            weights = weights_list
            diis = dii_list
            l1_loss_contributions = lassoterm_list
            l1_penalties = penalties
        l1_penalties = np.array(l1_penalties)

        self.history = {
            "l1_penalties": l1_penalties,
            "weights_per_l1_per_epoch": weights,
            "dii_per_l1_per_epoch": l1_penalties,
            "l1_term_per_l1_per_epoch": l1_loss_contributions,
        }

        (
            num_nonzero_features,
            l1_penalties_opt_per_nfeatures,
            dii_opt_per_nfeatures,
            weights_opt_per_nfeatures,
        ) = _extract_min_diis_lasso_optimization(weights, diis, l1_penalties)

        if plotlasso is True:
            _plot_min_lasso_results(
                dii_opt_per_nfeatures,
                num_nonzero_features,
                l1_penalties_opt_per_nfeatures,
            )

        return (
            num_nonzero_features,
            l1_penalties_opt_per_nfeatures,
            dii_opt_per_nfeatures,
            weights_opt_per_nfeatures,
        )
