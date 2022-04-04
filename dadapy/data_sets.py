# Copyright 2021 The DADApy Authors. All Rights Reserved.
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

import numpy as np

from dadapy.data import Data


class DataSets:
    def __init__(
        self,
        coordinates_list=(),
        distances_list=(),
        maxk_list=[None],
        verbose=False,
        njobs=1,
    ):
        if len(distances_list) == 0:
            self.N_sets = len(coordinates_list)
            distances_list = [None] * self.N_sets

        elif len(coordinates_list) == 0:
            self.N_sets = len(distances_list)
            coordinates_list = [None] * self.N_sets

        else:
            assert len(coordinates_list) == len(distances_list)
            self.N_sets = len(coordinates_list)

        if len(maxk_list) == 1:
            maxk_list = [maxk_list[0]] * self.N_sets

        self.data_sets = []

        for i in range(self.N_sets):
            X = coordinates_list[i]
            dists = distances_list[i]
            maxk = maxk_list[i]
            data = Data(
                coordinates=X, distances=dists, maxk=maxk, verbose=verbose, njobs=njobs
            )

            self.data_sets.append(data)

        # self.maxk = min([self.data_sets[i].maxk for i in range(self.N_sets)])  # maxk neighbourhoods

        self.verbose = verbose
        self.njobs = njobs
        self.ids = None  # ids
        self.ov_gt = None  # overlap ground truth (classes)
        self.ov_out = None  # overlap output neighborhoods
        self.ov_ll = None  # overlap ll neighbourhoods
        self.gamma = None  # gamma_matrix all to all

    def add_one_dataset(self, coordinates=None, distances=None, labels=None, maxk=None):

        data = Data(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            verbose=self.verbose,
            njobs=self.njobs,
        )

        self.data_sets.append(data)
        self.N_sets += 1

    def set_common_gt_labels(self, labels):

        for d in self.data_sets:
            assert d.gt_labels is not None

        for d in self.data_sets:
            d.gt_labels = labels

    def serialize_computation(self, computation_string, **kwargs):
        """This method applies the computation defined in 'computation_string' to all Data instances
        contained in the class.
        """
        # attr_dict = {'compute_id_2NN': 'ids',
        #             'compute_density_kNN': 'rhos',
        #             'compute_density_PAk': 'rhos'
        #             'compute_clustering': ''}

        for i, d in enumerate(self.data_sets):
            comput = getattr(d, computation_string)
            comput(**kwargs)

        # try:
        #     attr = getattr(self, attr_dict[computation_string])


if __name__ == "__main__":
    X1 = np.random.uniform(0, 1, (100, 2))
    X2 = np.random.uniform(0, 1, (100, 2))

    ds = DataSets(coordinates_list=([X1, X2]))

    ds.serialize_computation("compute_distances", maxk=3)

    print(ds.data_sets[0].distances)

    ds.serialize_computation("compute_clustering", Z=1)
