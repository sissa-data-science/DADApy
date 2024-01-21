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
The *data_sets* module contains the *DataSets* class.

Such a class is useful to manipulate multiple different datasets at the same time, and to compute specific metrics
between datasets, such as the information imbalance.
"""

import multiprocessing
import os

import numpy as np

from dadapy._utils.metric_comparisons import _return_imbalance
from dadapy._utils.utils import compute_nn_distances
from dadapy.data import Data

cores = multiprocessing.cpu_count()
np.set_printoptions(precision=2)
os.getcwd()


rng = np.random.default_rng(42)


class DataSets:
    """DataSets class."""

    def __init__(
        self,
    ):
        """Initialise a DataSets object.

        Args:
            ds (list(Data)): a list of "Data"-like objects
        """

    @staticmethod
    def return_inf_imb_jackknife_d1_to_d2(
        d1: Data, d2: Data, file_name="jackknife.txt", n=None, k=1
    ):
        """Return the mean and the standard deviation of the information imbalalce using the Jackknife method.

        Args:
            d1 (Data): The first dataset
            d2 (Data): The second dataset
            file_name (str, optional): The file where the imbalances are saved. Defaults to "jackknife.txt". Set to "None" to avoid saving.
            n (_type_, optional): Number of Jackknife repetitions. Defaults to the number of dataset points.
            k (int, optional): The number of neighbours for the imbalance computations. Defaults to 1.

        Returns:
            mean and standard deviation of the imbalance estimates from d1 to d2.
        """
        assert len(d1.X) == len(d2.X)
        if n == None:
            n = len(d1.X)

        if file_name is not None:
            print("Saving imbalances in " + file_name)
            file_object = open(file_name, "a")
        else:
            print("Not saving imbalances")

        random_indices = np.arange(len(d1.X))
        rng.shuffle(random_indices)

        removed_indices = random_indices[:n]
        imbalances_X1toX2 = []
        for i in removed_indices:
            print(i)
            X1_i = np.delete(d1.X, [i], axis=0)
            X2_i = np.delete(d2.X, [i], axis=0)
            _, d1_dist_indices = compute_nn_distances(
                X1_i, d1.maxk - 1, d1.metric, d1.period
            )  # remove 1 maxk for jackknife
            _, d2_dist_indices = compute_nn_distances(
                X2_i, d1.maxk - 1, d1.metric, d1.period
            )

            imb_X1toX2 = _return_imbalance(d1_dist_indices, d2_dist_indices, rng, k=k)
            imbalances_X1toX2.append(imb_X1toX2)

            if file_name is not None:
                np.savetxt(file_name, imbalances_X1toX2)

        mean, std = np.mean(imbalances_X1toX2), np.std(imbalances_X1toX2)

        return mean, std
