import time
import warnings

import cython
import numpy as np

cimport numpy as np
from libc.math cimport exp

DTYPE = np.int
floatTYPE = np.float
boolTYPE = np.bool

ctypedef np.int_t DTYPE_t
ctypedef np.float64_t floatTYPE_t

@cython.boundscheck(False)
@cython.cdivision(True)

def _compute_clustering(floatTYPE_t Z,
                        bint halo,
                        np.ndarray[DTYPE_t, ndim = 1] kstar,
                        np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                        DTYPE_t maxk,
                        bint verb,
                        np.ndarray[floatTYPE_t, ndim = 1] Rho_err,
                        np.ndarray[floatTYPE_t, ndim = 1] Rho_c,
                        np.ndarray[floatTYPE_t, ndim = 1] g,
                        DTYPE_t Nele):

    cdef DTYPE_t i, j, t, Nclus, check

    cdef DTYPE_t index, maxposidx

    cdef floatTYPE_t a1, a2, e1, e2, maxpos, lag, sec

    cdef DTYPE_t tmp, jmod, imod

    cdef DTYPE_t c, cp, k

    cdef DTYPE_t po, p2, pp, p1



    cdef np.ndarray[DTYPE_t, ndim = 1]  _centers_ = np.repeat(-1, Nele)
    cdef DTYPE_t len_centers = 0

    if verb: print("init succeded")
    sec = time.time()

# This for looks for the centers. A point is a center if its g is bigger than the one of all its neighbors
    for i in range(Nele):
        t = 0
        for j in range(1, kstar[i] + 1):
            if (g[i] < g[dist_indices[i, j]]):
                t = 1
                break
        if (t == 0):
            _centers_[len_centers] = i
            len_centers += 1

    cdef np.ndarray[DTYPE_t, ndim = 2]  to_remove = -np.ones((len_centers, len_centers), dtype=int)
    cdef DTYPE_t len_to_remove = 0
    cdef floatTYPE_t max_rho = -999.

    if verb:
      lag = time.time() - sec
      print(f"Raw identification of the putative centers: {lag: .3f} sec")
      sec = time.time()

# This  part  checks that there are no centers within the neighborhood of points with higher density.
    for k in range(len_centers):
        t=0
        max_rho = -999.
        for i in range(Nele):
            for j in range(1, kstar[i]+1):
                if (dist_indices[i, j] == _centers_[k]):
                    if (g[i] > g[_centers_[k]]):
                        to_remove[len_to_remove, 0] = _centers_[k]
                        t=1

                        if max_rho <0:
                            len_to_remove += 1

                        if g[i]>max_rho:
                            max_rho=g[i]
                            to_remove[len_to_remove, 1] = i

    if verb:
      lag = time.time() - sec
      print(f"Further checking on centers: {lag: .3f} sec ")
      sec = time.time()

    cdef np.ndarray[DTYPE_t, ndim = 1]  centers = np.empty(len_centers - len_to_remove, dtype=int)
    cdef DTYPE_t cindx = 0

    for i in range(len_centers):
        flag = 0
        for j in range(len_to_remove):
            if _centers_[i] == to_remove[j, 0]:
                flag = 1
                break
        if (flag == 0):
            centers[cindx] = _centers_[i]
            cindx += 1

    if verb:
      lag = time.time() - sec
      print(f"Pruning of the centers wrongly identified in part one: {lag: .3f} sec")
      sec = time.time()

    #the selected centers can't belong to the neighborhood of points with higher density
    cdef np.ndarray[DTYPE_t, ndim = 1]  cluster_init_ = -np.ones(Nele, dtype = int)
    Nclus = len_centers - len_to_remove

    for i in range(len_centers - len_to_remove):
        cluster_init_[centers[i]] = i


    sortg = np.argsort(-g)  # Rank of the elements in the g vector sorted in descendent order

    # Perform preliminar assignation to clusters
    maxk = dist_indices.shape[1]

    for j in range(Nele):
        ele = sortg[j]
        nn = 0
        while (cluster_init_[ele] == -1) and nn < maxk-1:
            nn = nn + 1
            cluster_init_[ele] = cluster_init_[dist_indices[ele, nn]]

        if cluster_init_[ele]==-1:
            ele_neighbors = dist_indices[ele]
            all_removed_centers = to_remove[:, 0]

            _, _, ind_removed_centers_ele = np.intersect1d(
                ele_neighbors, all_removed_centers, return_indices=True
            )

            # for i in range(all_removed_centers):
            #     for j in range(ele_neighbors):
            #         if ele_neighbors[j]==all_removed_centers[i]:

            # indices (in 'removed_centers') of the centers of higher density
            # in the neighborhood of 'removed centers'
            # (higher_density_centers --> centers of higher density that have a
            # 'removed center' in their neighborhood)
            higher_density_centers = to_remove[:, 1]
            higher_density_centers_ele = higher_density_centers[
                ind_removed_centers_ele
            ]

            # index (in 'cluster_init') of the 'maximum density center' of such neighboring centers
            max_center = higher_density_centers_ele[
                np.argmax(g[higher_density_centers_ele])
            ]

            # new_highest_neighbor = index_highest_nb_density[index_highest_nb_density]
            cluster_init_[ele] = cluster_init_[max_center]

    clstruct = []  # useful list of points in the clusters
    for i in range(Nclus):
        x1 = []
        for j in range(Nele):
            if (cluster_init_[j] == i):
                x1.append(j)
        clstruct.append(x1)


    if verb:
      lag = time.time() - sec
      print(f"Preliminary assignation finished: {lag: .3f} sec")
      print("Number of clusters before multimodality test=", Nclus)
      sec = time.time()


    #this implementation can be vry costly if Nclus is > 10^4
    if Nclus > 10000:
        warnings.warn(
        """There are > 10k initial putative clusters:
        the matrices of the saddle points may cause out of memory error (6x Nclus x Nclus --> > 4.8 GB required).
        If this is the case, call compute_clustering_ADP_pure_python(v2 = True). Ignore the warning otherwise."""
        )

    cdef np.ndarray[floatTYPE_t, ndim = 2]  Rho_bord = np.zeros((Nclus, Nclus))
    cdef np.ndarray[floatTYPE_t, ndim = 2]  Rho_bord_err = np.zeros((Nclus, Nclus))
    cdef np.ndarray[DTYPE_t, ndim = 2]  Point_bord = np.zeros((Nclus, Nclus), dtype=int)

    cdef np.ndarray[floatTYPE_t, ndim = 1]  pos = np.zeros(Nclus * Nclus)
    cdef np.ndarray[DTYPE_t, ndim = 1]  ipos = np.zeros(Nclus * Nclus, dtype=int)
    cdef np.ndarray[DTYPE_t, ndim = 1]  jpos = np.zeros(Nclus * Nclus, dtype=int)

    cdef np.ndarray[DTYPE_t, ndim = 1]  cluster_init = np.array(cluster_init_)



    # Find border points between putative clusters

    sec = time.time()

    for i in range(Nclus):
        for j in range(Nclus):
            Point_bord[i, j] = -1

    #
    # i and j are points belonging to the clusters c(i) and c(j)
    #
    # Point i is a border point between c(i) and c(j) if:
    #                   a) It has in its neighborhood a point j belonging to other cluster.
    #                   b) There are no other points belonging to c(i) nearer from point j
    #                   c) It has the maximum density among the points that fulfill a) & b)
    #

    for c in range(Nclus):
        for p1 in clstruct[c]:
            if p1 in centers:
                pp=-1
            else:
                for k in range(1, kstar[p1] + 1):
                    p2 = dist_indices[p1, k]
                    pp = -1
                    if (cluster_init[p2] != c):
                        pp = p2
                        cp = cluster_init[pp]
                        break
            if (pp != -1):
                for k in range(1, maxk):
                    po = dist_indices[pp, k]
                    if (po == p1):
                        break
                    if (cluster_init[po] == c):
                        pp = -1
                        break
            if (pp != -1):
                if (g[p1] > Rho_bord[c, cp]):
                    Rho_bord[c, cp] = g[p1]
                    Rho_bord[cp, c] = g[p1]
                    Point_bord[cp, c] = p1
                    Point_bord[c, cp] = p1

    # Symmetrize matrix
    for i in range(Nclus - 1):
        for j in range(i + 1, Nclus):
            if (Point_bord[i, j] != -1):
                Rho_bord[i, j] = Rho_c[Point_bord[i, j]]
                Rho_bord[j, i] = Rho_c[Point_bord[j, i]]
                Rho_bord_err[i, j] = Rho_err[Point_bord[i, j]]
                Rho_bord_err[j, i] = Rho_err[Point_bord[j, i]]

    for i in range(Nclus):
        Rho_bord[i, i] = -1.
        Rho_bord_err[i, i] = 0.


    if verb:
      lag = time.time() - sec
      print(f"Identification of the saddle points: {lag: .3f} sec")
      sec = time.time()

    check = 1
    sec = time.time()

    cdef np.ndarray[DTYPE_t, ndim = 1]  centers_ = np.array(centers, dtype=int)
    cdef np.ndarray[DTYPE_t, ndim = 1]  clsurv = np.ones(Nclus, dtype=int)

    # sec = time.time()
    # Here we start the merging process through multimodality test.
    secp = 0

    while (check == 1):

        check = 0

        for i in range(Nclus * Nclus):
            pos[i] = 0.0
            ipos[i] = 0
            jpos[i] = 0

        index = 0
        maxposidx = 0
        maxpos = - 9999999999

        for i in range(Nclus - 1):
            for j in range(i + 1, Nclus):
                a1 = (Rho_c[centers_[i]] - Rho_bord[i, j])
                a2 = (Rho_c[centers_[j]] - Rho_bord[i, j])
                e1 = Z * (Rho_err[centers_[i]] + Rho_bord_err[i, j])
                e2 = Z * (Rho_err[centers_[j]] + Rho_bord_err[i, j])
                if (a1 < e1 or a2 < e2):
                    check = 1
                    pos[index] = Rho_bord[i, j]
                    ipos[index] = i
                    jpos[index] = j
                    if pos[index] > maxpos:
                        maxpos = pos[index]
                        maxposidx = index

                    index = index + 1

        if (check == 1):
            barriers = maxposidx

            imod = ipos[barriers]
            jmod = jpos[barriers]
            c1=(Rho_c[centers[imod]] - Rho_bord[imod][jmod])/(Rho_err[centers[imod]] + Rho_bord_err[imod][jmod])
            c2=(Rho_c[centers[jmod]] - Rho_bord[imod][jmod])/(Rho_err[centers[jmod]] + Rho_bord_err[imod][jmod])
            if c1<c2:

                tmp = jmod
                jmod = imod
                imod = tmp

            clsurv[jmod] = 0

            Rho_bord[imod, jmod] = -1.
            Rho_bord[jmod, imod] = -1.
            Rho_bord_err[imod, jmod] = 0.
            Rho_bord_err[jmod, imod] = 0.

            clstruct[imod].extend(clstruct[jmod])

            clstruct[jmod] = []

            for i in range(Nclus):
                if (i != imod and i != jmod):
                    if (Rho_bord[imod, i] < Rho_bord[jmod, i]):

                        Point_bord[imod, i] = Point_bord[jmod, i]
                        Point_bord[i, imod] = Point_bord[imod, i]

                        Rho_bord[imod, i] = Rho_bord[jmod, i]
                        Rho_bord[i, imod] = Rho_bord[imod, i]

                        Rho_bord_err[imod, i] = Rho_bord_err[jmod, i]
                        Rho_bord_err[i, imod] = Rho_bord_err[imod, i]

                    Rho_bord[jmod, i] = -1
                    Rho_bord[i, jmod] = Rho_bord[jmod, i]
                    Rho_bord_err[jmod, i] = 0
                    Rho_bord_err[i, jmod] = Rho_bord_err[jmod, i]


    if verb:
      lag = time.time() - sec
      print(f"Multimodality test finished: {lag: .3f} sec")
      sec = time.time()

    Nclus_m = 0
    clstruct_m = []
    centers_m = []
    nnum = []
    for j in range(Nclus):
        nnum.append(-1)
        if (clsurv[j] == 1):
            nnum[j] = Nclus_m
            Nclus_m = Nclus_m + 1
            clstruct_m.append(clstruct[j])
            centers_m.append(centers[j])

    Point_bord_m = np.zeros((Nclus_m, Nclus_m), dtype=int)
    Rho_bord_err_m = np.zeros((Nclus_m, Nclus_m), dtype=float)
    Rho_bord_m = np.zeros((Nclus_m, Nclus_m), dtype=float)

    for j in range(Nclus):
        if (clsurv[j] == 1):
            jj = nnum[j]
            for k in range(Nclus):
                if (clsurv[k] == 1):
                    kk = nnum[k]
                    Rho_bord_m[jj][kk] = Rho_bord[j][k]
                    Rho_bord_err_m[jj][kk] = Rho_bord_err[j][k]
                    Point_bord_m[jj][kk]=Point_bord[j][k]

    Last_cls = np.empty(Nele, dtype=int)
    for j in range(Nclus_m):
        for k in clstruct_m[j]:
            Last_cls[k] = j
    Last_cls_halo = np.copy(Last_cls)
    nh = 0
    for j in range(Nclus_m):
        Rho_halo = max(Rho_bord_m[j])
        for k in clstruct_m[j]:
            if (Rho_c[k] < Rho_halo):
                nh = nh + 1
                Last_cls_halo[k] = -1
    if (halo):
        labels = Last_cls_halo
    else:
        labels = Last_cls

    out_bord = np.copy(Rho_bord_m)

    if verb:
      lag = time.time() - sec
      print(f"Final operations: {lag} sec")
      sec = time.time()

    return clstruct_m, Nclus_m, labels, centers_m, out_bord, Rho_bord_err_m, Point_bord_m
