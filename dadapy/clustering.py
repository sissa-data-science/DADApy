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

"""
The *clustering* module contains the *Clustering* class.

Density-based clustering algorithms are implemented as methods of this class.
"""

import multiprocessing
import time

import numpy as np
import scipy as sp

from dadapy._cython import cython_clustering as cf
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
        self, coordinates=None, distances=None, maxk=None, verbose=False, njobs=cores
    ):
        """Initialise the Clustering class."""
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            verbose=verbose,
            njobs=njobs,
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

    def compute_clustering_ADP(
        self, Z=1.65, halo=False, density_algorithm="PAk", k=None, Dthr=23.92812698
    ):
        """Compute clustering according to the algorithm DPA.

        The only free parameter is the merging factor Z, which controls how the different density peaks are merged
        together. The higher the Z, the more aggressive the merging, the smaller the number of clusters.
        The calculation is optimized though cython

        Args:
            Z(float): merging parameter
            halo (bool): compute (or not) the halo points
            density_algorithm (str): method to compute the local density.
                Use 'PAK' for adaptive neighbourhood, 'kNN' for fixed neighbourhood
            k (int): number of neighbours when using kNN algorithm
            Dthr (float): Likelihood ratio parameter used to compute optimal k when using PAK algorithm.
                The value of Dthr=23.92 corresponds to a p-value of 1e-6.

        References:
            M. d’Errico, E. Facco, A. Laio, A. Rodriguez, Automatic topography  of  high-dimensional  data  sets  by
                non-parametric  density peak clustering, Information Sciences 560 (2021) 476–492

        """
        if self.log_den is None:

            if density_algorithm == "PAk":
                self.compute_density_PAk(Dthr=Dthr)

            elif density_algorithm == "kNN":
                assert k is not None, "provide k to estimate the density with kNN"
                self.compute_density_kNN(k=k)

            else:
                raise NameError('density estimators name must be "PAK" or "kNN" ')

        if self.verb:
            print("Clustering started")

        # Make all values of log_den positives (this is important to help convergence)
        # even when substracting the value Z*log_den_err
        log_den_min = np.min(self.log_den - Z * self.log_den_err)

        log_den_c = self.log_den - log_den_min + 1

        # Putative modes of the PDF as preliminary clusters
        g = log_den_c - self.log_den_err

        # centers are point of max density  (max(g) ) within their optimal neighborhood (defined by kstar)
        seci = time.time()

        out = cf._compute_clustering(
            Z,
            halo,
            self.kstar,
            self.dist_indices.astype(int),
            self.maxk,
            self.verb,
            self.log_den_err,
            log_den_min,
            log_den_c,
            g,
            self.N,
        )

        secf = time.time()

        self.cluster_indices = out[0]
        self.N_clusters = out[1]
        self.cluster_assignment = out[2]
        self.cluster_centers = out[3]
        out_bord = out[4]
        log_den_min = out[5]
        self.log_den_bord_err = out[6]
        self.bord_indices = out[7]

        self.log_den_bord = out_bord + log_den_min - 1 - np.log(self.N)

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
        self, Z=1.65, halo=False, density_algorithm="PAk", k=None
    ):
        """Compute ADP clustering, but without the cython optimization."""
        if self.log_den is None:

            if density_algorithm == "PAk":
                self.compute_density_PAk()

            elif density_algorithm == "kNN":
                assert k is not None, "provide k to estimate the density with kNN"
                self.compute_density_kNN(k=k)

            else:
                raise NameError('density estimators name must be "PAK" or "kNN" ')

        if self.verb:
            print("Clustering started")

        # Make all values of log_den positives (this is important to help convergence)
        # even when substracting the value Z*log_den_err
        log_den_min = np.min(self.log_den - Z * self.log_den_err)
        log_den_c = self.log_den - log_den_min + 1

        # Putative modes of the PDF as preliminary clusters
        sec = time.time()
        N = self.N
        g = log_den_c - self.log_den_err
        centers = []
        for i in range(N):
            t = 0
            for j in range(1, self.kstar[i] + 1):
                if g[i] < g[self.dist_indices[i, j]]:
                    t = 1
                    break
            if t == 0:
                centers.append(i)

        count = 0
        centers_iter = centers.copy()
        for i in centers_iter:
            l, m = np.where(self.dist_indices == i)
            for j in range(l.shape[0]):
                if (g[l[j]] > g[i]) & (m[j] <= self.kstar[l[j]]):
                    centers.remove(i)
                    count += 1
                    break

        cluster_init = []
        for _ in range(N):
            cluster_init.append(-1)
            Nclus = len(centers)
        if self.verb:
            print("Number of clusters before multimodality test=", Nclus)

        for i in centers:
            cluster_init[i] = centers.index(i)
        sortg = np.argsort(-g)  # Get the rank of the elements in the g vector
        # sorted in descendent order.
        # Perform preliminar assignation to clusters
        for j in range(N):
            ele = sortg[j]
            nn = 0
            while cluster_init[ele] == -1:
                nn = nn + 1
                cluster_init[ele] = cluster_init[self.dist_indices[ele, nn]]
        clstruct = []  # useful list of points in the clusters
        for i in range(Nclus):
            x1 = []
            for j in range(N):
                if cluster_init[j] == i:
                    x1.append(j)
            clstruct.append(x1)
        sec2 = time.time()
        if self.verb:
            print(
                "{0:0.2f} seconds clustering before multimodality test".format(
                    sec2 - sec
                )
            )
        log_den_bord = np.zeros((Nclus, Nclus), dtype=float)
        log_den_bord_err = np.zeros((Nclus, Nclus), dtype=float)
        Point_bord = np.zeros((Nclus, Nclus), dtype=int)
        # Find border points between putative clusters
        sec = time.time()
        for i in range(Nclus):
            for j in range(Nclus):
                Point_bord[i][j] = -1
        for c in range(Nclus):
            for p1 in clstruct[c]:
                if p1 in centers:
                    pp = -1
                else:
                    for k in range(1, self.kstar[p1] + 1):
                        p2 = self.dist_indices[p1, k]
                        pp = -1
                        if cluster_init[p2] != c:
                            pp = p2
                            cp = cluster_init[pp]
                            break
                if pp != -1:
                    for k in range(1, self.maxk):
                        po = self.dist_indices[pp, k]
                        if po == p1:
                            break
                        if cluster_init[po] == c:
                            pp = -1
                            break
                if pp != -1:
                    if g[p1] > log_den_bord[c][cp]:
                        log_den_bord[c][cp] = g[p1]
                        log_den_bord[cp][c] = g[p1]
                        Point_bord[cp][c] = p1
                        Point_bord[c][cp] = p1

        for i in range(Nclus - 1):
            for j in range(i + 1, Nclus):
                if Point_bord[i][j] != -1:
                    log_den_bord[i][j] = log_den_c[Point_bord[i][j]]
                    log_den_bord[j][i] = log_den_c[Point_bord[j][i]]
                    log_den_bord_err[i][j] = self.log_den_err[Point_bord[i][j]]
                    log_den_bord_err[j][i] = self.log_den_err[Point_bord[j][i]]
        for i in range(Nclus):
            log_den_bord[i][i] = -1.0
            log_den_bord_err[i][i] = 0.0
        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds identifying the borders".format(sec2 - sec))
        check = 1
        clsurv = []
        sec = time.time()
        for _ in range(Nclus):
            clsurv.append(1)
        while check == 1:
            pos = []
            ipos = []
            jpos = []
            check = 0
            for i in range(Nclus - 1):
                for j in range(i + 1, Nclus):
                    a1 = log_den_c[centers[i]] - log_den_bord[i][j]
                    a2 = log_den_c[centers[j]] - log_den_bord[i][j]
                    e1 = Z * (self.log_den_err[centers[i]] + log_den_bord_err[i][j])
                    e2 = Z * (self.log_den_err[centers[j]] + log_den_bord_err[i][j])
                    if a1 < e1 or a2 < e2:
                        check = 1
                        pos.append(log_den_bord[i][j])
                        ipos.append(i)
                        jpos.append(j)
            if check == 1:
                barriers = pos.index(max(pos))
                imod = ipos[barriers]
                jmod = jpos[barriers]
                c1 = (log_den_c[centers[imod]] - log_den_bord[imod][jmod]) / (
                    self.log_den_err[centers[imod]] + log_den_bord_err[imod][jmod]
                )
                c2 = (log_den_c[centers[jmod]] - log_den_bord[imod][jmod]) / (
                    self.log_den_err[centers[jmod]] + log_den_bord_err[imod][jmod]
                )
                if c1 < c2:
                    tmp = jmod
                    jmod = imod
                    imod = tmp
                clsurv[jmod] = 0
                log_den_bord[imod][jmod] = -1.0
                log_den_bord[jmod][imod] = -1.0
                log_den_bord_err[imod][jmod] = 0.0
                log_den_bord_err[jmod][imod] = 0.0
                clstruct[imod].extend(clstruct[jmod])
                clstruct[jmod] = []
                for i in range(Nclus):
                    if i != imod and i != jmod:
                        if log_den_bord[imod][i] < log_den_bord[jmod][i]:
                            log_den_bord[imod][i] = log_den_bord[jmod][i]
                            log_den_bord[i][imod] = log_den_bord[imod][i]
                            log_den_bord_err[imod][i] = log_den_bord_err[jmod][i]
                            log_den_bord_err[i][imod] = log_den_bord_err[imod][i]
                        log_den_bord[jmod][i] = -1
                        log_den_bord[i][jmod] = log_den_bord[jmod][i]
                        log_den_bord_err[jmod][i] = 0
                        log_den_bord_err[i][jmod] = log_den_bord_err[jmod][i]
        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds with multimodality test".format(sec2 - sec))
        N_clusters = 0
        cluster_indices = []
        cluster_centers = []
        nnum = []
        for j in range(Nclus):
            nnum.append(-1)
            if clsurv[j] == 1:
                nnum[j] = N_clusters
                N_clusters = N_clusters + 1
                cluster_indices.append(clstruct[j])
                cluster_centers.append(centers[j])
        Point_bord_m = np.zeros((N_clusters, N_clusters), dtype=int)
        log_den_bord_m = np.zeros((N_clusters, N_clusters), dtype=float)
        log_den_bord_err_m = np.zeros((N_clusters, N_clusters), dtype=float)
        for j in range(Nclus):
            if clsurv[j] == 1:
                jj = nnum[j]
                for k in range(Nclus):
                    if clsurv[k] == 1:
                        kk = nnum[k]
                        log_den_bord_m[jj][kk] = log_den_bord[j][k]
                        log_den_bord_err_m[jj][kk] = log_den_bord_err[j][k]
                        Point_bord_m[jj][kk] = Point_bord[j][k]
        Last_cls = np.empty(N, dtype=int)
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
        out_bord = np.copy(log_den_bord_m)
        self.cluster_indices = cluster_indices
        self.N_clusters = N_clusters
        self.cluster_assignment = cluster_assignment
        self.cluster_centers = cluster_centers
        self.log_den_bord = (
            out_bord + log_den_min - 1
        )  # remove wrong normalisation introduced earlier
        self.log_den_bord_err = log_den_bord_err_m
        self.bord_indices = Point_bord_m
        if self.verb:
            print("Clustering finished, {} clusters found".format(self.N_clusters))

        return self.cluster_assignment
