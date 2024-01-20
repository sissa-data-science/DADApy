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
The *clustering* module contains the *Clustering* class.

Density-based clustering algorithms are implemented as methods of this class.
"""

import multiprocessing
import time
import warnings

import numpy as np
import scipy as sp

from dadapy._cython import cython_clustering as cf
from dadapy._cython import cython_clustering_v2 as cf2
from dadapy.density_estimation import DensityEstimation

cores = multiprocessing.cpu_count()


class Clustering(DensityEstimation):
    """Perform clustering using various density-based clustering algorithms.

    Inherits from the DensityEstimation class.

    Attributes:
        N_clusters (int): Number of clusters found
        cluster_assignment (list(int)): A list of length N containing the cluster assignment of each point as an
            integer from 0 to N_clusters-1.
        cluster_centers (list(int)): Indices of the centroids of each cluster (density peak)
        cluster_indices (list(list(int))): A list of lists. Each sublist contains the indices belonging to the
            corresponding cluster.
        log_den_bord (np.ndarray(float)): A matrix of dimensions N_clusters x N_clusters containing
            the estimated log density of the saddle point between each couple of peaks.
        log_den_bord_err (np.ndarray(float)): A matrix of dimensions N_clusters x N_clusters containing
            the estimated error on the log density of the saddle point between each couple of peaks.
        bord_indices (np.ndarray(float)): A matrix of dimensions N_clusters x N_clusters containing the indices of
            the saddle point between each couple of peaks.

    """

    def __init__(
        self, coordinates=None, distances=None, maxk=None, verbose=False, n_jobs=cores
    ):
        """Initialise the Clustering class."""
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            verbose=verbose,
            n_jobs=n_jobs,
        )

        self.cluster_indices = None
        self.N_clusters = None
        self.cluster_assignment = None
        self.cluster_centers = None
        self.log_den_bord_err = None
        self.log_den_bord = None
        self.bord_indices = None

        self.delta = None  # Minimum distance from an element with higher density
        self.ref = None  # Index of the nearest element with higher density

    def compute_clustering_ADP(self, Z=1.65, halo=False, v2=False):
        """Compute clustering according to the algorithm DPA.

        The only free parameter is the merging factor Z, which controls how the different density peaks are merged
        together. The higher the Z, the more aggressive the merging, the smaller the number of clusters.
        The calculation is optimized though cython

        Args:
            Z(float): merging parameter
            halo (bool): compute (or not) the halo points

        Returns:
            cluster_assignment (np.ndarray(int)): assignment of points to specific clusters

        References:
            M. d’Errico, E. Facco, A. Laio, A. Rodriguez, Automatic topography  of  high-dimensional  data  sets  by
                non-parametric  density peak clustering, Information Sciences 560 (2021) 476–492

        """
        if self.log_den is None:
            self.compute_density_PAk()

        assert not np.isnan(np.sum(self.log_den)), "log density contains nan values"
        assert not np.isnan(
            np.sum(self.log_den_err)
        ), "log error density contains nan values"

        if self.verb:
            print("Clustering started")

        if v2 is True:
            warnings.warn(
                """using adp implementation v2: this requires less memory but can
                be two times slower than the original implementation""",
                stacklevel=2,
            )

        # Make all values of log_den positives (this is important to help convergence)
        # even when subtracting the value Z*log_den_err
        log_den_min = np.min(self.log_den - Z * self.log_den_err)
        log_den_c = self.log_den - log_den_min + 1

        # Putative modes of the PDF as preliminary clusters
        g = log_den_c - self.log_den_err

        # centers are point of max density  (max(g) ) within their optimal neighborhood (defined by kstar)
        seci = time.time()

        if v2:
            out = cf2._compute_clustering(
                Z,
                halo,
                self.kstar,
                self.dist_indices.astype(int),
                self.maxk,
                self.verb,
                self.log_den_err,
                log_den_c,
                g,
                self.N,
            )
        else:
            out = cf._compute_clustering(
                Z,
                halo,
                self.kstar,
                self.dist_indices.astype(int),
                self.maxk,
                self.verb,
                self.log_den_err,
                log_den_c,
                g,
                self.N,
            )

        secf = time.time()

        self.cluster_indices = out[0]
        self.N_clusters = out[1]
        self.cluster_assignment = out[2]
        self.cluster_centers = out[3]
        self.log_den_bord = out[4] + log_den_min - 1
        self.log_den_bord_err = out[5]
        self.bord_indices = out[6]

        if self.verb:
            print(f"Clustering finished, {self.N_clusters} clusters found")
            print(f"total time is, {secf - seci}")

        return self.cluster_assignment

    def compute_DecGraph(self):
        """Compute the decision graph."""
        assert self.log_den is not None, "Compute density before"
        assert self.X is not None
        self.delta = np.zeros(self.N)
        self.ref = np.zeros(self.N, dtype="int")
        tt = np.arange(self.N)
        imax = []
        ncalls = 0
        for i in range(self.N):
            ll = tt[((self.log_den > self.log_den[i]) & (tt != i))]
            if ll.shape[0] > 0:
                a1, a2, a3 = np.intersect1d(
                    self.dist_indices[i, :], ll, return_indices=True, assume_unique=True
                )
                if a1.shape[0] > 0:
                    aa = np.min(a2)
                    self.delta[i] = self.distances[i, aa]
                    self.ref[i] = self.dist_indices[i, aa]
                else:
                    ncalls = ncalls + 1
                    dd = self.X[((self.log_den > self.log_den[i]) & (tt != i))]
                    ds = np.transpose(
                        sp.spatial.distance.cdist([np.transpose(self.X[i, :])], dd)
                    )
                    j = np.argmin(ds)
                    self.ref[i] = ll[j]
                    self.delta[i] = ds[j]
            else:
                self.delta[i] = -100.0
                imax.append(i)
        self.delta[imax] = 1.05 * np.max(self.delta)
        print("Number of points for which self.delta needed call to cdist=", ncalls)

    def compute_clustering_DP(self, dens_cut=0.0, delta_cut=0.0, halo=False):
        """Compute clustering using the Density Peak algorithm.

        Args:
            dens_cut (float): cutoff on density values
            delta_cut (float): cutoff on distance values
            halo (bool): use or not halo points

        Returns:
            cluster_assignment (np.ndarray(int)): assignment of points to specific clusters

        References:
            A. Rodriguez, A. Laio, Clustering by fast search and find of density peaks,
            Science 344 (6191) (2014) 1492–1496.
        """
        assert self.delta is not None
        ordered = np.argsort(-self.log_den)
        self.cluster_assignment = np.zeros(self.N, dtype="int")
        tt = np.arange(self.N)
        center_label = np.zeros(self.N, dtype="int")
        ncluster = -1
        for i in range(self.N):
            j = ordered[i]
            if (self.log_den[j] > dens_cut) & (self.delta[j] > delta_cut):
                ncluster = ncluster + 1
                self.cluster_assignment[j] = ncluster
                center_label[j] = ncluster
            else:
                self.cluster_assignment[j] = self.cluster_assignment[self.ref[j]]
                center_label[j] = -1
        self.centers = tt[(center_label != -1)]
        if halo:
            bord = np.zeros(self.N, dtype="int")
            halo = np.copy(self.cluster_assignment)

            for i in range(self.N):
                for j in self.dist_indices[i, :][(self.distances[i, :] <= self.dc[i])]:
                    if self.cluster_assignment[i] != self.cluster_assignment[j]:
                        bord[i] = 1
            halo_cutoff = np.zeros(ncluster + 1)
            halo_cutoff[:] = np.min(self.log_den) - 1
            for j in range(ncluster + 1):
                td = self.log_den[((bord == 1) & (self.cluster_assignment == j))]
                if td.size != 0:
                    halo_cutoff[j] = np.max(td)
            halo[tt[(self.log_den < halo_cutoff[self.cluster_assignment])]] = -1
            self.cluster_assignment = halo

        return self.cluster_assignment

    def compute_clustering_ADP_pure_python(  # noqa: C901
        self, Z=1.65, halo=False, v2=False
    ):
        """Compute ADP clustering, but without the cython optimization."""
        if self.log_den is None:
            self.compute_density_PAk()

        assert not np.isnan(np.sum(self.log_den)), "log density contains nan values"
        assert not np.isnan(
            np.sum(self.log_den_err)
        ), "log error density contains nan values"

        if self.verb:
            print("Clustering started")

        if v2 is True:
            warnings.warn(
                """using adp implementation v2: this requires less memory but
                can be two times slower than the original implementation""",
                stacklevel=2,
            )

        # Make all values of log_den positives (this is important to help convergence)
        # even when subtracting the value Z*log_den_err
        log_den_min = np.min(self.log_den - Z * self.log_den_err)
        # define a "log_den_c" specific to perform clustering
        log_den_c = self.log_den - log_den_min + 1

        # Find the putative modes of the PDF as preliminary clusters
        sec = time.time()

        g = log_den_c - self.log_den_err

        centers, removed_centers = self._find_density_modes(g)

        Nclus = len(centers)

        if self.verb:
            print("Number of clusters before multimodality test=", Nclus)

        cluster_init, cl_struct = self._preliminary_cluster_assignment(
            g, centers, removed_centers
        )

        sec2 = time.time()
        if self.verb:
            print(
                "{0:0.2f} seconds clustering before multimodality test".format(
                    sec2 - sec
                )
            )

        # Find border points between putative clusters
        sec = time.time()
        if not v2:
            (
                log_den_bord,
                log_den_bord_err,
                bord_index,
            ) = self._find_borders_between_clusters(
                Nclus, g, cl_struct, centers, cluster_init, log_den_c
            )
        else:
            saddle_density, saddle_indices = self._find_borders_between_clusters_v2(
                Nclus, g, cl_struct, centers, cluster_init, log_den_c
            )

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds identifying the borders".format(sec2 - sec))

        sec = time.time()
        if not v2:
            surviving_clusters, bord_index = self._multimodality_test(
                Nclus,
                Z,
                log_den_c,
                centers,
                cl_struct,
                log_den_bord,
                log_den_bord_err,
                bord_index,
            )
        else:
            (
                surviving_clusters,
                saddle_density,
                saddle_indices,
            ) = self._multimodality_test_v2(
                Nclus, Z, log_den_c, centers, cl_struct, saddle_density, saddle_indices
            )

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds with multimodality test".format(sec2 - sec))

        if not v2:
            (
                N_clusters,
                cluster_assignment,
                cluster_indices,
                cluster_centers,
                log_den_bord,
                log_den_bord_err,
                bord_indices,
            ) = self._finalise_clustering(
                Nclus,
                halo,
                surviving_clusters,
                cl_struct,
                centers,
                log_den_bord,
                log_den_bord_err,
                bord_index,
                log_den_c,
            )

        else:
            (
                N_clusters,
                cluster_assignment,
                cluster_indices,
                cluster_centers,
                log_den_bord,
                log_den_bord_err,
                bord_indices,
            ) = self._finalise_clustering_v2(
                Nclus,
                halo,
                surviving_clusters,
                cl_struct,
                centers,
                saddle_density,
                saddle_indices,
                log_den_c,
            )

        self.cluster_indices = cluster_indices
        self.N_clusters = N_clusters
        self.cluster_assignment = cluster_assignment
        self.cluster_centers = cluster_centers
        self.log_den_bord = (
            log_den_bord + log_den_min - 1
        )  # remove wrong normalisation introduced earlier
        self.log_den_bord_err = log_den_bord_err
        self.bord_indices = bord_indices

        if self.verb:
            print("Clustering finished, {} clusters found".format(self.N_clusters))

        return self.cluster_assignment

    # ------------ helper methods for compute_clustering_ADP_pure_python ------------ #

    def _find_density_modes(self, g):
        """Find the modes of the density."""
        centers = []
        for i in range(self.N):
            t = 0
            for j in range(1, self.kstar[i] + 1):
                if g[i] < g[self.dist_indices[i, j]]:
                    t = 1
                    break

            # "i" is a center if it has no point at a higher density
            if t == 0:
                centers.append(i)

        centers_iter = centers.copy()

        removed_centers = []
        for i_center in centers_iter:
            l, m = np.where(self.dist_indices == i_center)

            # keep only neighborhoods where i_center is within kstar
            mask = m <= self.kstar[l]
            l, m = l[mask], m[mask]
            # index of the point of maximum density in these neighborhoods
            max_rho = np.argmax(g[l])
            max_rho = l[max_rho]

            # if the density of 'max_rho' is higher than that of i_center,
            # remove i_center and store [i_center,  max_rho] (see later)
            if g[max_rho] > g[i_center]:
                # check if this max_rho is already in removed centers
                if len(removed_centers) > 0:
                    is_valid = np.where(np.array(removed_centers)[:, 0] == max_rho)[0]
                    # if current max_rho is in removed centers use its max_rho
                    if is_valid.size > 0:
                        max_rho = removed_centers[is_valid[0]][1]

                removed_centers.append([i_center, max_rho])
                centers.remove(i_center)

        return centers, np.array(removed_centers)

    def _preliminary_cluster_assignment(self, g, centers, removed_centers):
        """Find a preliminary assignment of points to the closest density peak.

        Args:
            g: scaled log density of points
            centers: preliminary density peaks

        Returns:
            cluster_init (list(int)): preliminary assignations of points to clusters
            cl_struct (list(list(int))): list of points in each cluster
        """
        # all points assigned to no clusters initially
        cluster_init = [-1] * self.N

        # assign centers to their own cluster
        for i in centers:
            cluster_init[i] = centers.index(i)

        # Get the rank of the elements in the g vector
        # sorted in decreasing order.
        sortg = np.argsort(-g)

        # Perform preliminary assignation to clusters
        maxk = self.dist_indices.shape[1]
        for j in range(self.N):
            ele = sortg[j]
            nn = 0
            while cluster_init[ele] == -1 and nn < maxk - 1:
                nn = nn + 1
                cluster_init[ele] = cluster_init[self.dist_indices[ele, nn]]

            # if in ther first maxk there is no point assigned to a cluster,
            # assign the point to the nearby center of max density
            if cluster_init[ele] == -1:
                # all the neighbors of ele (ele included!)
                ele_neighbors = self.dist_indices[ele]

                # indices (in 'removed_centers') of removed centers in the neighbor of ele (may include ele itself!)
                # (all_removed_centers --> indices of the centers that have been removed)
                all_removed_centers = removed_centers[:, 0]

                _, _, ind_removed_centers_ele = np.intersect1d(
                    ele_neighbors, all_removed_centers, return_indices=True
                )

                # indices (in 'removed_centers') of the centers of higher density
                # in the neighborhood of 'removed centers'
                # (higher_density_centers --> centers of higher density that have a
                # 'removed center' in their neighborhood)
                higher_density_centers = removed_centers[:, 1]
                higher_density_centers_ele = higher_density_centers[
                    ind_removed_centers_ele
                ]

                # index (in 'cluster_init') of the 'maximum density center' of such neighboring centers
                max_center = higher_density_centers_ele[
                    np.argmax(g[higher_density_centers_ele])
                ]

                cluster_init[ele] = cluster_init[max_center]

        # useful list of points in the clusters
        cl_struct = []
        for i in range(len(centers)):
            x1 = []
            for j in range(self.N):
                if cluster_init[j] == i:
                    x1.append(j)
            cl_struct.append(x1)

        return cluster_init, cl_struct

    def _find_borders_between_clusters(  # noqa: C901
        self, Nclus, g, cl_struct, centers, cluster_init, log_den_c
    ):
        """Find saddle points between clusters.

        Assume i and j are points belonging to the clusters c(i) and c(j).
        Then, point i is a border point between c(i) and c(j) if:
            a) It has in its neighborhood a point j belonging to other cluster.
            b) There are no other points belonging to c(i) nearer from point j
            c) It has the maximum density among the points that fulfill a) & b)

        Args:
            Nclus: number of clusters
            g: scaled log density of points
            cl_struct: points assigned to each cluster
            centers: density peaks
            cluster_init: cluster assignment
            log_den_c: log density of points

        Returns:
            log_den_bord: log density of the saddle points
            log_den_bord_err: error on the log density of saddle points
            bord_index: square matrix providing the index of the saddle point between any two peaks
        """
        # this implementation can be vry costly if Nclus is > 10^4
        if Nclus > 10000:
            warnings.warn(
                """There are > 10k initial putative clusters:
            the matrices of the saddle points may cause out of memory error (3x Nclus x Nclus --> > 2.4 GB required).
            If this is the case, call compute_clustering_ADP_pure_python(v2 = True). Ignore the warning otherwise.""",
                stacklevel=2,
            )

        try:
            log_den_bord = np.zeros((Nclus, Nclus), dtype=np.float64)
            log_den_bord_err = np.zeros((Nclus, Nclus), dtype=np.float64)
            # set all bord indices to -1 (no points) initially
            bord_index = np.ones((Nclus, Nclus), dtype=int) * -1
        except MemoryError:
            print(
                f"Nclus = {Nclus}. Out of memory error: call compute_clustering_ADP_pure_python(v2 = True)"
            )

        for c in range(Nclus):
            for p1 in cl_struct[c]:  # p1 points in a given cluster
                if p1 in centers:
                    pp = -1

                else:
                    # a point p2 in the neighborhood of p1 belongs to a cluster cp different from p1
                    for k in range(1, self.kstar[p1] + 1):
                        p2 = self.dist_indices[p1, k]
                        pp = -1
                        if (
                            cluster_init[p2] != c
                        ):  # a point in the neighborhood of p1 belongs to another cluster
                            pp = p2
                            cp = cluster_init[pp]  # neighbor cluster index
                            break
                if pp != -1:
                    for k in range(1, self.maxk):
                        po = self.dist_indices[pp, k]  # pp here is p2
                        if po == p1:  # p1 is the closest point of p2 belonging to c
                            break
                        if (
                            cluster_init[po] == c
                        ):  # there is a point of c closest to p2 than p1
                            pp = -1
                            break

                # here check if the border point is a saddle point
                if pp != -1:
                    if (
                        g[p1] > log_den_bord[c][cp]
                    ):  # g = log_den_c - self.log_den_err different from next log_den_bord without log_den_err
                        log_den_bord[c][cp] = g[p1]
                        log_den_bord[cp][c] = g[p1]
                        bord_index[cp][c] = p1
                        bord_index[c][cp] = p1

        # fill in matrices of bord densities and errors in a symmetric way
        for i in range(Nclus - 1):
            for j in range(i + 1, Nclus):
                if bord_index[i][j] != -1:
                    log_den_bord[i][j] = log_den_c[bord_index[i][j]]
                    log_den_bord[j][i] = log_den_c[bord_index[j][i]]
                    log_den_bord_err[i][j] = self.log_den_err[bord_index[i][j]]
                    log_den_bord_err[j][i] = self.log_den_err[bord_index[j][i]]

        # set diagonal to -1
        for i in range(Nclus):
            log_den_bord[i][i] = -1.0
            log_den_bord_err[i][i] = 0.0

        return log_den_bord, log_den_bord_err, bord_index

    def _multimodality_test(
        self,
        Nclus,
        Z,
        log_den_c,
        centers,
        cl_struct,
        log_den_bord,
        log_den_bord_err,
        bord_index,
    ):
        """Merge couples of peaks if these are not statistically significant."""
        check = 1

        # all clusters are initialised as "surviving"
        surviving_clusters = [1] * Nclus

        while check == 1:
            # density and position of borders to be merged
            pos = []
            ipos = []
            jpos = []

            check = 0

            # check whether there are statistically not-significant couples of peaks
            for i in range(Nclus - 1):
                for j in range(i + 1, Nclus):
                    # differences in peaks i->j and j->i
                    a1 = log_den_c[centers[i]] - log_den_bord[i][j]
                    a2 = log_den_c[centers[j]] - log_den_bord[i][j]

                    # statistical thresholds
                    e1 = Z * (self.log_den_err[centers[i]] + log_den_bord_err[i][j])
                    e2 = Z * (self.log_den_err[centers[j]] + log_den_bord_err[i][j])

                    # if two peaks are not statistically significant, save their border
                    if a1 < e1 or a2 < e2:
                        check = 1
                        pos.append(log_den_bord[i][j])
                        ipos.append(i)
                        jpos.append(j)

            # merging process
            if check == 1:
                # start merging from border of higher density
                barrier_index = pos.index(max(pos))
                imod = ipos[barrier_index]
                jmod = jpos[barrier_index]

                # normalised log density of first and second peak to be merged
                c1 = (log_den_c[centers[imod]] - log_den_bord[imod][jmod]) / (
                    self.log_den_err[centers[imod]] + log_den_bord_err[imod][jmod]
                )
                c2 = (log_den_c[centers[jmod]] - log_den_bord[imod][jmod]) / (
                    self.log_den_err[centers[jmod]] + log_den_bord_err[imod][jmod]
                )

                # by default, the second peak is removed (c2) if however c1 < c2 the indices are
                # exchanged and the first cluster is removed instead
                if c1 < c2:
                    tmp = jmod
                    jmod = imod
                    imod = tmp

                # cluster jmod is removed
                surviving_clusters[jmod] = 0
                # log_den_bord between the removed peaks is set to -1 and error to 0
                log_den_bord[imod][jmod] = -1.0
                log_den_bord[jmod][imod] = -1.0
                log_den_bord_err[imod][jmod] = 0.0
                log_den_bord_err[jmod][imod] = 0.0

                # the points of the jmod peak are added to the imod peak
                cl_struct[imod].extend(cl_struct[jmod])
                cl_struct[jmod] = []

                # recompute the borders with the other peaks
                for i in range(Nclus):
                    if i != imod and i != jmod:
                        if log_den_bord[imod][i] < log_den_bord[jmod][i]:
                            bord_index[imod][i] = bord_index[jmod][i]
                            bord_index[i][imod] = bord_index[imod][i]

                            log_den_bord[imod][i] = log_den_bord[jmod][i]
                            log_den_bord[i][imod] = log_den_bord[imod][i]

                            log_den_bord_err[imod][i] = log_den_bord_err[jmod][i]
                            log_den_bord_err[i][imod] = log_den_bord_err[imod][i]

                        log_den_bord[jmod][i] = -1
                        log_den_bord[i][jmod] = log_den_bord[jmod][i]
                        log_den_bord_err[jmod][i] = 0
                        log_den_bord_err[i][jmod] = log_den_bord_err[jmod][i]

        return surviving_clusters, bord_index

    def _finalise_clustering(  # noqa: C901
        self,
        Nclus,
        halo,
        surviving_clusters,
        cl_struct,
        centers,
        log_den_bord,
        log_den_bord_err,
        bord_index,
        log_den_c,
    ):
        """Finalise clustering."""
        # compute final N_clusters, cluster_indices and cluster_centers
        N_clusters = 0
        cluster_indices = []
        cluster_centers = []
        nnum = []
        for j in range(Nclus):
            nnum.append(-1)
            if surviving_clusters[j] == 1:
                nnum[j] = N_clusters
                N_clusters += 1
                cluster_indices.append(cl_struct[j])
                cluster_centers.append(centers[j])

        # initialise the final arrays
        bord_indices_m = np.zeros((N_clusters, N_clusters), dtype=int)
        log_den_bord_m = np.zeros((N_clusters, N_clusters), dtype=float)
        log_den_bord_err_m = np.zeros((N_clusters, N_clusters), dtype=float)

        for j in range(Nclus):
            if surviving_clusters[j] == 1:
                jj = nnum[j]
                for k in range(Nclus):
                    if surviving_clusters[k] == 1:
                        kk = nnum[k]
                        log_den_bord_m[jj][kk] = log_den_bord[j][k]
                        log_den_bord_err_m[jj][kk] = log_den_bord_err[j][k]
                        bord_indices_m[jj][kk] = bord_index[j][k]

        Last_cls = np.empty(self.N, dtype=int)
        for j in range(N_clusters):
            for k in cluster_indices[j]:
                Last_cls[k] = j
        Last_cls_halo = np.copy(Last_cls)
        nh = 0
        for j in range(N_clusters):
            log_den_halo = max(log_den_bord_m[j])
            for k in cluster_indices[j]:
                if log_den_c[k] < log_den_halo:
                    nh = nh + 1
                    Last_cls_halo[k] = -1
        if halo:
            cluster_assignment = Last_cls_halo
        else:
            cluster_assignment = Last_cls

        log_den_bord_m = np.copy(log_den_bord_m)

        return (
            N_clusters,
            cluster_assignment,
            cluster_indices,
            cluster_centers,
            log_den_bord_m,
            log_den_bord_err_m,
            bord_indices_m,
        )

    def _find_borders_between_clusters_v2(  # noqa: C901
        self, Nclus, g, cl_struct, centers, cluster_init, log_den_c
    ):
        """Find saddle points between clusters.

        Assume i and j are points belonging to the clusters c(i) and c(j).
        Then, point i is a border point between c(i) and c(j) if:
            a) It has in its neighborhood a point j belonging to other cluster.
            b) There are no other points belonging to c(i) nearer from point j
            c) It has the maximum density among the points that fulfill a) & b)

        Args:
            Nclus: number of clusters                       int
            g: scaled log density of points                 np.array(N)
            cl_struct: points assigned to each cluster      list(list())
            centers: data index of density peaks            np.array(Nclus)
            cluster_init: cluster assignment                np.array(N)
            log_den_c: log density of points                np.array(N)

        Returns:
            log_den_bord: log density of the saddle points
            log_den_bord_err: error on the log density of saddle points
            bord_index: square matrix providing the index of the saddle point between any two peaks
        """
        saddle_count = 0
        for c in range(Nclus):
            # saddle point index, saddle point density, saddle point error, cluster1 index, cluster2 index, valid_saddle
            # create another array of indices

            saddle_density_tmp = np.zeros(
                (Nclus, 3), dtype=float
            )  # density, density_error, normalized_density
            saddle_indices_tmp = -np.ones(
                (Nclus, 4), dtype=int
            )  # saddle point, cluster1, cluster2, is_valid saddle

            if c == 0:
                saddle_density = saddle_density_tmp
                saddle_indices = saddle_indices_tmp
                # array of saddle points of cluster  c (will be eventually much less than Nclus)
            else:
                saddle_density = np.concatenate(
                    (saddle_density, saddle_density_tmp), axis=0
                )
                saddle_indices = np.concatenate(
                    (saddle_indices, saddle_indices_tmp), axis=0
                )

            for p1 in cl_struct[c]:  # p1 point in a given cluster
                if p1 in centers:
                    pp = -1

                else:
                    # a point p2 in the neighborhood of p1 belongs to a cluster cp different from p1
                    for k in range(1, self.kstar[p1] + 1):
                        p2 = self.dist_indices[p1, k]
                        pp = -1
                        if (
                            cluster_init[p2] != c
                        ):  # a point in the neighborhood of p1 belongs to another cluster
                            pp = p2
                            cp = cluster_init[pp]  # neighbor cluster index
                            break

                if pp != -1:
                    # p1 is the closest point in c to p2
                    for k in range(1, self.maxk):  # why maxk?
                        po = self.dist_indices[pp, k]  # pp here is p2
                        if po == p1:  # p1 is the closest point of p2 belonging to c
                            break
                        if (
                            cluster_init[po] == c
                        ):  # p1 is NOT the closest point of p2 belonging to c
                            pp = -1
                            break

                # here check if the border point is a saddle point
                if pp != -1:
                    # maybe contrunct a matrix of border points [p, g, c1, c2]
                    # in log_den_bord_tmp the cluster are coupled with indices that go from smaller to bigger
                    if c > cp:
                        c1 = cp
                        c2 = c
                    else:
                        c1 = c
                        c2 = cp

                    flag = 0
                    for i in range(saddle_count):
                        if c1 == saddle_indices[i, 1] and c2 == saddle_indices[i, 2]:
                            flag = 1  # there is already a border point between c and cp
                            if g[p1] > saddle_density[i, 2]:
                                saddle_indices[i, 0] = p1  # useful at the end
                                saddle_density[i] = np.array(
                                    [log_den_c[p1], self.log_den_err[p1], g[p1]]
                                )
                                break

                    if flag == 0:
                        saddle_indices[saddle_count] = np.array(
                            [p1, c1, c2, 1], dtype=int
                        )
                        saddle_density[saddle_count] = np.array(
                            [log_den_c[p1], self.log_den_err[p1], g[p1]]
                        )
                        saddle_count += 1

            saddle_density = saddle_density[:saddle_count]
            saddle_indices = saddle_indices[:saddle_count]

        sort_indices = saddle_density[:, 0].argsort()[::-1]
        saddle_density = saddle_density[sort_indices]
        saddle_indices = saddle_indices[sort_indices]

        return saddle_density[:, :2], saddle_indices

    def _multimodality_test_v2(
        self, Nclus, Z, log_den_c, centers, cl_struct, saddle_density, saddle_indices
    ):
        """Merge couples of peaks if these are not statistically significant.

        Args:
            Nclus: number of clusters                       int
            Z: scaled log density of points                 int
            log_den_c: log density of points                np.array(N)
            centers: data index of density peaks            np.array(Nclus)
            cl_struct: points assigned to each cluster      list(list())
            log_den_bord,                                   np.array((Nclus, 6))
        """
        check = 1
        # all clusters are initialised as "surviving"
        surviving_clusters = np.ones(Nclus, dtype=int)
        nsaddles = len(saddle_density)

        while check == 1:
            check = 0
            current_saddle = -1.0
            for i in range(nsaddles):  # log_den_bord already sorted
                if saddle_indices[i, 3] == 1:
                    clus1, clus2 = (
                        saddle_indices[i, 1],
                        saddle_indices[i, 2],
                    )  # cluster indices
                    center1, center2 = (
                        centers[clus1],
                        centers[clus2],
                    )  # data indices peak of clus1, clus2

                    a1 = (
                        log_den_c[center1] - saddle_density[i, 0]
                    )  # log_den_c is an array of the density of all the points
                    a2 = log_den_c[center2] - saddle_density[i, 0]
                    sum_err1 = self.log_den_err[center1] + saddle_density[i, 1]
                    sum_err2 = self.log_den_err[center2] + saddle_density[i, 1]

                    # statistical thresholds
                    e1 = Z * sum_err1
                    e2 = Z * sum_err2
                    if a1 < e1 or a2 < e2:
                        check = 1
                        if current_saddle < saddle_density[i, 0]:
                            max_a1, max_a2 = a1, a2
                            max_sum_err1, max_sum_err2 = sum_err1, sum_err2
                            to_remove = i
                            current_saddle = saddle_density[i, 0]

            if check == 1:
                saddle_indices[
                    to_remove, -1
                ] = 0  # the couple center1, center2 is removed
                margin1 = max_a1 / max_sum_err1
                margin2 = max_a2 / max_sum_err2

                # only the peak with the highest margin is kept:
                # by convention we set this peak to be center1
                if margin1 < margin2:
                    c1, c2 = saddle_indices[to_remove, 2], saddle_indices[to_remove, 1]
                else:
                    c1, c2 = saddle_indices[to_remove, 1], saddle_indices[to_remove, 2]

                surviving_clusters[c2] = 0  # the second peak is removed
                cl_struct[c1].extend(cl_struct[c2])
                cl_struct[c2] = []

                # select the clusters that are neighbors to cluster1 and cluster2
                # cluster label, saddle label, position of c1/c2 in saddle_indices[:, 1:3]

                saddle_indices = self._find_neighbor_clusters_v2(
                    saddle_density, saddle_indices, to_remove, nsaddles, c1, c2
                )

        return surviving_clusters, saddle_density, saddle_indices

    def _finalise_clustering_v2(  # noqa: C901
        self,
        Nclus,
        halo,
        surviving_clusters,
        cl_struct,
        centers,
        saddle_density,
        saddle_indices,
        log_den_c,
    ):
        """Finalise clustering."""
        N_clusters = 0
        mapping = -np.ones(Nclus, dtype=int)  # all original centers
        cluster_indices, cluster_centers = [], []
        for i in range(Nclus):
            if surviving_clusters[i] == 1:
                mapping[i] = N_clusters  # center is surviving
                N_clusters += 1
                # !!!!!! check the index i is correct?
                cluster_indices.append(cl_struct[i])
                cluster_centers.append(centers[i])

        bord_indices_m = -np.ones((N_clusters, N_clusters), dtype=int)
        log_den_bord_m = np.zeros((N_clusters, N_clusters), dtype=float)
        log_den_bord_err_m = np.zeros((N_clusters, N_clusters), dtype=float)

        for i in range(len(saddle_density)):  # all original saddles
            if saddle_indices[i, 3] == 1:  # a valid saddle is between two valid centers
                j, k = (
                    mapping[saddle_indices[i, 1]],
                    mapping[saddle_indices[i, 2]],
                )  # map the two valid centers in the corresponding valid new cluster labels
                log_den_bord_m[j, k] = saddle_density[i, 0]
                log_den_bord_m[k, j] = log_den_bord_m[j, k]

                log_den_bord_err_m[j, k] = saddle_density[i, 1]
                log_den_bord_err_m[k, j] = log_den_bord_err_m[j, k]

                bord_indices_m[j, k] = saddle_indices[i, 0]
                bord_indices_m[k, j] = bord_indices_m[j, k]

        log_den_bord_m[np.diag_indices(N_clusters)] = -1

        Last_cls = np.empty(self.N, dtype=int)
        for j in range(N_clusters):
            for k in cluster_indices[j]:
                Last_cls[k] = j
        Last_cls_halo = np.copy(Last_cls)
        nh = 0
        for j in range(N_clusters):
            log_den_halo = max(log_den_bord_m[j])
            for k in cluster_indices[j]:
                if log_den_c[k] < log_den_halo:
                    nh = nh + 1
                    Last_cls_halo[k] = -1
        if halo:
            cluster_assignment = Last_cls_halo
        else:
            cluster_assignment = Last_cls

        log_den_bord_m = np.copy(log_den_bord_m)

        return (
            N_clusters,
            cluster_assignment,
            cluster_indices,
            cluster_centers,
            log_den_bord_m,
            log_den_bord_err_m,
            bord_indices_m,
        )

    def _find_neighbor_clusters_v2(  # noqa: C901
        self,
        saddle_density,
        saddle_indices,
        to_remove,
        nsaddles,
        c1,
        c2,
    ):
        neighbors1 = np.zeros((len(saddle_density), 3), dtype=int)
        neighbors2 = np.zeros((len(saddle_density), 3), dtype=int)
        len_neigh1, len_neigh2 = 0, 0
        for i in range(nsaddles):
            if i != to_remove and saddle_indices[i, 3] == 1:
                cluster_couple = saddle_indices[i, 1:3]

                if c1 in cluster_couple:
                    neighbors1[len_neigh1] = np.array([cluster_couple[0], i, 2])
                    if cluster_couple[0] == c1:
                        neighbors1[len_neigh1] = np.array([cluster_couple[1], i, 1])
                    len_neigh1 += 1

                elif c2 in cluster_couple:
                    neighbors2[len_neigh2] = np.array([cluster_couple[0], i, 2])
                    if cluster_couple[0] == c2:
                        neighbors2[len_neigh2] = np.array([cluster_couple[1], i, 1])
                    len_neigh2 += 1

        neighbors1, neighbors2 = neighbors1[:len_neigh1], neighbors2[:len_neigh2]

        # check which are the common neighbors between cluster1 and cluster2
        neighbors1 = neighbors1[neighbors1[:, 0].argsort()]
        neighbors2 = neighbors2[neighbors2[:, 0].argsort()]
        i, j = 0, 0
        while i < len(neighbors1) and j < len(neighbors2):
            if neighbors1[i, 0] == neighbors2[j, 0]:  # there is a common element
                index1, index2 = (
                    neighbors1[i, 1],
                    neighbors2[j, 1],
                )  # index common neighbor w.r.t cluster1
                if saddle_density[index1, 0] < saddle_density[index2, 0]:
                    # if the density of the saddle is higher between cluster2 and common neighbor,
                    # use this as new saddle between cluster1 and common neighbor
                    saddle_density[index1, 0] = saddle_density[
                        index2, 0
                    ]  # saddle density
                    saddle_density[index1, 1] = saddle_density[
                        index2, 1
                    ]  # saddle error
                # the couples of common elements with the c2 are "deleted"
                saddle_indices[index2, -1] = 0
                i += 1
                j += 1

            elif neighbors1[i, 0] < neighbors2[j, 0]:
                i += 1
            else:
                # neighbors of c2 which were not neighbors of c1 become neighbors of c1
                index2, c2_index = neighbors2[j, 1], neighbors2[j, 2]
                saddle_indices[index2, c2_index] = c1
                j += 1

        # some js in the comparison may not have been taken into account
        while j < len(neighbors2):
            index2, c2_index = neighbors2[j, 1], neighbors2[j, 2]
            saddle_indices[index2, c2_index] = c1
            j += 1

        return saddle_indices
