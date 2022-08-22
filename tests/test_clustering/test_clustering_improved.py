# Copyright 2021-2022 The DADApy Authors. All Rights Reserved.
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

"""Module for testing clustering methods ADP (original/v2), pure python (original/v2)"""

import os

import numpy as np

from dadapy import data

# ground_truths computed with 5000k datapoints of dataset ../../examples/datasets/Fig1.dat
path = os.path.join(os.path.split(__file__)[0], "./ground_truths")

nclusters_gt = np.load(f"{path}/nclusters5k.npy")
assignment_gt = np.load(f"{path}/assignment5k.npy")
centers_gt = np.load(f"{path}/centers5k.npy")
bord_indices_gt = np.load(f"{path}/bord_indices5k.npy")
saddle_density_gt = np.load(f"{path}/saddle_density5k.npy")
saddle_err_gt = np.load(f"{path}/saddle_err5k.npy")


path = os.path.join(os.path.split(__file__)[0], "../../examples/datasets")
X = np.genfromtxt(f"{path}/Fig1.dat")
X = X[:5000]
cl = data.Data(coordinates=X)
_ = cl.compute_clustering_ADP(v2=False)
nclusters_adp = cl.N_clusters
assignment_adp = cl.cluster_assignment
centers_adp = np.array(cl.cluster_centers)
saddle_density_adp = cl.log_den_bord
saddle_err_adp = cl.log_den_bord_err
bord_indices_adp = cl.bord_indices

X = np.genfromtxt(f"{path}/Fig1.dat")
X = X[:5000]
cl = data.Data(coordinates=X)
_ = cl.compute_clustering_ADP(v2=True)
nclusters_adp_v2 = cl.N_clusters
assignment_adp_v2 = cl.cluster_assignment
centers_adp_v2 = np.array(cl.cluster_centers)
saddle_density_adp_v2 = cl.log_den_bord
saddle_err_adp_v2 = cl.log_den_bord_err
bord_indices_adp_v2 = cl.bord_indices


X = np.genfromtxt(f"{path}/Fig1.dat")
X = X[:5000]
cl = data.Data(coordinates=X)
_ = cl.compute_clustering_ADP_pure_python(v2=False)
nclusters_pp = cl.N_clusters
assignment_pp = cl.cluster_assignment
centers_pp = np.array(cl.cluster_centers)
saddle_density_pp = cl.log_den_bord
saddle_err_pp = cl.log_den_bord_err
bord_indices_pp = cl.bord_indices


X = np.genfromtxt(f"{path}/Fig1.dat")
X = X[:5000]
cl = data.Data(coordinates=X)
_ = cl.compute_clustering_ADP_pure_python(v2=True)
nclusters_pp_v2 = cl.N_clusters
assignment_pp_v2 = cl.cluster_assignment
centers_pp_v2 = np.array(cl.cluster_centers)
saddle_density_pp_v2 = cl.log_den_bord
saddle_err_pp_v2 = cl.log_den_bord_err
bord_indices_pp_v2 = cl.bord_indices

# CHECK CONSISTENCY CLUSTERING ATTRIBUTES

# ADP consistent with ground truth
assert nclusters_adp == nclusters_gt
assert np.all(assignment_adp == assignment_gt)
assert np.all(centers_adp == centers_gt)
assert np.all(bord_indices_adp == bord_indices_gt)
assert np.allclose(saddle_density_adp, saddle_density_gt)
assert np.allclose(saddle_err_adp, saddle_err_gt)

# ADPv2 consisten with ADP
assert nclusters_adp == nclusters_adp_v2
assert np.all(assignment_adp == assignment_adp_v2)
assert np.all(centers_adp == centers_adp_v2)
assert np.all(bord_indices_adp == bord_indices_adp_v2)
assert np.allclose(saddle_density_adp, saddle_density_adp_v2)
assert np.allclose(saddle_err_adp, saddle_err_adp_v2)

# pure python consisten with ADP
assert nclusters_adp == nclusters_pp
assert np.all(assignment_adp == assignment_pp)
assert np.all(centers_adp == centers_pp)
assert np.all(bord_indices_adp == bord_indices_pp)
assert np.allclose(saddle_density_adp, saddle_density_pp)
assert np.allclose(saddle_err_adp, saddle_err_pp)

# pure python v2 consisten with ADP
assert nclusters_adp == nclusters_pp_v2
assert np.all(assignment_adp == assignment_pp_v2)
assert np.all(centers_adp == centers_pp_v2)
assert np.all(bord_indices_adp == bord_indices_pp_v2)
assert np.allclose(saddle_density_adp, saddle_density_pp_v2)
assert np.allclose(saddle_err_adp, saddle_err_pp_v2)


# np.save('nclusters5k.npy', nclusters_adp)
# np.save('assignment5k.npy', assignment_adp)
# np.save('centers5k.npy', centers_adp)
# np.save('bord_indices5k.npy', bord_indices_adp)
# np.save('saddle_density5k.npy', saddle_density_adp)
# np.save('saddle_err5k.npy', saddle_err_adp)
