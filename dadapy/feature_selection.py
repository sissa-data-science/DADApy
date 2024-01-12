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
The *feature_selection* module contains the *FeatureSelection* class.

This class uses Differentiable Information Imbalance
"""

import multiprocessing
import warnings
from typing import Type

import numpy as np

from dadapy.base import Base
from dadapy.metric_comparisons import MetricComparisons
from dadapy._utils.metric_comparisons import _return_ranks
from dadapy._utils.utils import compute_nn_distances
from dadapy._utils.differentiable_imbalance import (
    _compute_kernel_imbalance,
    _compute_kernel_imbalance_gradient,
    _compute_full_dist_matrix,
    _compute_full_rank_matrix,
    _optimize_learning_rate,
)

cores = multiprocessing.cpu_count()

class FeatureSelection(MetricComparisons):
    def __init__(
        self,
        coordinates=None,
        distances=None,
        maxk=None,
        period=None,
        verbose=False,
        njobs=cores,
    ):
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            period=period,
            verbose=verbose,
            njobs=njobs,
        )

        # TODO: This does not need to be here
        self.cythond = True
        # TODO: Stop returning history everywhere and instead save it here and implement self.get_history()
        self.history = None 

    @staticmethod
    def _check_maxk(maxk, ndata):
        # TODO: Remove once kernel imbalance works for maxk
        if maxk != ndata-1:
            warnings.warn(
                f"""maxk neighbors is not available for this functionality.\
                It will be ignored and treated as the number of data-1, {ndata}""", 
                stacklevel=2
            )
    
    def _check_own_maxk(self):
        self._check_maxk(self.maxk, self.coordinates.shape[0])

    @staticmethod
    def compute_optimal_lambda(distance_matrix, fraction=1.):
        # TODO: consider most likely use case and stop having it accept a distance matrix
        # TODO: if kept like this move to _utils.differentiable_imbalance
        # sets lambda to the average between the smallest and mean (2nd NN - 1st NN)-distance
        # np.fill_diagonal(distance_matrix, np.nan) ###CHANGE: This I don't need because on the diagonal I have just big values
        NNs = np.sort(distance_matrix, axis=1) #
        min_distances_nn = NNs[:,1] - NNs[:,0]
        return fraction * ((np.min(min_distances_nn) + np.nanmean(min_distances_nn)) / 2)

    def compute_kernel_imbalance(self, target_data: Type[Base], lambd=None):
        """Computes the kernel imbalance between two matrices based on distances of input data and rank information of groundtruth data.

        Args:
            dist_matrix_A (np.ndarray): N x N array - The distance matrix for between all input data points of input space A. Can
                be computed with 'compute_dist_matrix'
            rank_matrix_B (np.ndarray): N x N rank matrix for the groundtruth data B. 
            lambd (float, optional): The regularization parameter. Default: 0.1. The higher this value, the more nearest neighbors are included. 
                Can be calculated automatically with 'compute_optimal_lambda'. This sets lambda to a distance smaller than the average distance 
                in the data set but bigger than the minimal distance

        Returns:
            kernel_imbalance (float): The computed kernel imbalance value. Please mind that this depends (unlike in the classic information imbalance) 
                on the chosen lambda. To compare several scaled distances compare them using the same lambda, even if they were optimized with different ones.

        Raises:
            None.
        """
        # TODO: only accepting target data of Base (or children) is slow if base automatically calculates distances.
        # TODO: either add lazyBase or find other way to implement things like period and metric of Base.
        if lambd is None:
            lambd = self.compute_optimal_lambda(self.X)

        distances_i = _compute_full_dist_matrix(self.X, period=self.period, n_jobs=self.njobs)
        rank_matrix_j = _compute_full_rank_matrix(target_data.X, period=target_data.period, n_jobs=self.njobs)

        return _compute_kernel_imbalance(dist_matrix_i=distances_i, rank_matrix_j=rank_matrix_j, lambd=lambd)
    
        # sets lambda to the average between the smallest and mean (2nd NN - 1st NN)-distance
    
    def compute_kernel_imbalance_gradient(self, target_data: Type[Base], gammas: np.ndarray, lambd: float=None):
        if lambd is None:
            lambd = self.compute_optimal_lambda(self.X)
        
        rescaled_distances_i = _compute_full_dist_matrix(self.X*gammas, period=self.period, n_jobs=self.njobs)
        rank_matrix_j = _compute_full_rank_matrix(target_data.X, period=target_data.period, n_jobs=self.njobs)

        return _compute_kernel_imbalance_gradient(rescaled_distances_i, self.X, rank_matrix_j, gammas=gammas, lambd=lambd, period=self.period, n_jobs=self.njobs)

    def optimize_kernel_imbalance(
            self, target_data: Type[Base], n_epochs=100, constrain=False,
            initial_gammas: np.ndarray[float]=None, lambd: float=None,
            learning_rate: float=None, l1_penalty=0., decaying_lr=True
        ):
        # TODO: do typechecks here, maybe remove some functions above
        raise NotImplementedError()
        return _compute_kernel_imbalance_gradient()
        
    def eliminate_backward_greedy_kernel_imbalance(self, groundtruth_data, data, gammas_0, lambd=None, n_epochs=100, l_rate=0.1, constrain=False, decaying_lr=True, period=None, groundtruthperiod=None, n_jobs=None):
        """Do a stepwise backward eliminitaion of features and after each elimination GD otpmize the kernel imblance
        Args:
            groundtruth_data (np.ndarray): N x D(groundtruth) array containing N datapoints in all the groundtruth features D(groundtruth)
            data (np.ndarray): N x D(input) array containing N datapoints in D input features whose weights are optimized to reproduce the groundtruth distances
            gammas_0 (np.ndarray or list): D(input) initial weights for the input features. No zeros allowed here
            lambd (float): softmax scaling. If None (preferred) this chosen automatically with compute_optimial_lambda
            n_epochs (int): number of epochs in each optimization cycle
            l_rate (float): learning rate. Has to be tuned, especially if constrain=True (otherwise optmization could fail)
            constrain (bool): if True, rescale the weights so the biggest weight = 1
            l1_penalty (float): l1 regularization strength
            decaying_lr (bool): default: True. Apply decaying learning rate = l_rate * 2**(-i_epoch/10) - every 10 epochs the learning rate will be halfed
            period (float or np.ndarray/list): D(input) periods (input formatted to be 0-period). If not a list, the same period is assumed for all D features
            groundtruthperiod (float or np.ndarray/list): D(groundtruth) periods (groundtruth formatted to be 0-period). If not a list, the same period is assumed for all D(groundtruth) features
        Returns:
            gammas_list (np.ndarray): D x n_epochs x D. All weights for each optimization step for each number of nonzero weights. For final weights: gammas_list[:,-1,:]
            kernel_imbalances_list (np.ndarray): D x n_epochs. Imbalance for each optimization step for each number of nonzero weights. For final imbalances: kernel_imbalances_list[:,-1] 
            """
        # find a suitable learning rate by chosing the best optimization
        if l_rate == None:
            l_rate, _ = optimize_learning_rate(groundtruth_data=groundtruth_data, data=data, gammas_0=gammas_0, lambd=lambd, 
                            n_epochs=50, constrain=False, l1_penalty=0.0, decaying_lr=decaying_lr, 
                            period=period, groundtruthperiod=groundtruthperiod, nsamples=300, lr_list=None)

        gammaslist=[]
        imbalancelist=[]
        #do this just for making a warm start even for the first optimization
        gammass, imbalances, _ = optimize_kernel_imbalance(groundtruth_data=groundtruth_data, data=data, gammas_0=gammas_0,
                                                        lambd=lambd, n_epochs=n_epochs, l_rate=l_rate, constrain=constrain,
                                                        l1_penalty=0., decaying_lr=decaying_lr, period=period, groundtruthperiod=groundtruthperiod, 
                                                        n_jobs=n_jobs, cythond=self.cythond)
        gammaslist.append(gammass)
        imbalancelist.append(imbalances)

        gammasss=gammass[-1]
        nonzeros = linalg.norm(gammasss,0)
        counter=len(gammasss)

        while nonzeros >= 1:
            start = time.time()
            gs, imbs = optimize_kernel_imbalance_static_zeros(groundtruth_data=groundtruth_data, data=data, gammas_0=gammasss, lambd=lambd, 
                                                            n_epochs=n_epochs, l_rate=l_rate, constrain=constrain, decaying_lr=decaying_lr, 
                                                            period=period, groundtruthperiod=groundtruthperiod, n_jobs=n_jobs, cythond=self.cythond)

            end = time.time()
            timing = end - start
            print("number of nonzero weights= ", nonzeros, ", time: ", timing)
            gammasss=gs[-1]
            arr=1*gammasss
            arr[arr == 0] = np.nan
            if np.isnan(arr).all():
                gammaslist.append(gs)
                imbalancelist.append(imbs)
                break
            mingamma = np.nanargmin(arr)
            gammasss[mingamma] = 0
            nonzeros = linalg.norm(gammasss,0)
            gammaslist.append(gs)
            imbalancelist.append(imbs)


        return np.array(gammaslist), np.array(imbalancelist)
    
    def search_lasso_optimization_kernel_imbalance(self, groundtruth_data, data, gammas_0, lambd=None, n_epochs=100, 
                                                l_rate=None, constrain=False, decaying_lr=True, period=None, 
                                                groundtruthperiod=None, n_jobs=None):
        # Initial l1 search 
        
        if l_rate == None:
            l_rate, _ = optimize_learning_rate(groundtruth_data=groundtruth_data, data=data, gammas_0=gammas_0, lambd=lambd,
                            n_epochs=39, constrain=constrain, l1_penalty=0.0, decaying_lr=decaying_lr,
                            period=period, groundtruthperiod=groundtruthperiod, nsamples=300, lr_list=None)

        l1_penalties = [0] + list(np.linspace((1/l_rate)/200, (1/l_rate)*2, 9)) # test l1's depending on the learning rate
        
        gs = np.zeros((len(l1_penalties),n_epochs+1,data.shape[1]))
        ks = np.zeros((len(l1_penalties),n_epochs+1))
        ls = np.zeros((len(l1_penalties),n_epochs+1))

        for i in range(len(l1_penalties)):
            gs[i], ks[i], ls[i] = optimize_kernel_imbalance(groundtruth_data=groundtruth_data, data=data, gammas_0=gammas_0, lambd=lambd,
                                                            n_epochs=n_epochs, l_rate=l_rate, constrain=constrain, l1_penalty=l1_penalties[i], 
                                                            decaying_lr=decaying_lr, period=period, groundtruthperiod=groundtruthperiod, n_jobs=n_jobs, cythond=self.cythond)
        
        # Refine l1 search
        
        gammas_list, kernel_list, lassoterm_list, penalties = refine_lasso_optimization(gs, ks, ls, l1_penalties, groundtruth_data=groundtruth_data, 
                                                                            data=data, gammas_0=gammas_0, lambd=lambd, n_epochs=n_epochs, l_rate=l_rate, 
                                                                            constrain=constrain, decaying_lr=decaying_lr, 
                                                                            period=period, groundtruthperiod=groundtruthperiod, n_jobs=n_jobs, cythond=self.cythond)
        
        return gammas_list, kernel_list, lassoterm_list, penalties