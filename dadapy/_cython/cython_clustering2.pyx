import time

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
                        floatTYPE_t Rho_min,
                        np.ndarray[floatTYPE_t, ndim = 1] Rho_c,
                        np.ndarray[floatTYPE_t, ndim = 1] g,
                        DTYPE_t Nele):

    cdef DTYPE_t i, j, t, Nclus, check

    cdef DTYPE_t index, maxposidx

    cdef floatTYPE_t a1, a2, e1, e2, maxpos

    cdef DTYPE_t tmp, jmod, imod

    cdef DTYPE_t c, cp, k

    cdef DTYPE_t po, p2, pp, p1

    sec = time.time()

    cdef np.ndarray[DTYPE_t, ndim = 1]  _centers_ = np.repeat(-1, Nele)
    cdef DTYPE_t len_centers = 0

    if verb: print("init succeded")

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
        print("part one finished: Raw identification of the putative centers")

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
        print("part two finished: Further checking on centers")

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
        print("part tree finished: Pruning of the centers wrongly identified in part one.")


    #the selected centers can't belong to the neighborhood of points with higher density
    cdef np.ndarray[DTYPE_t, ndim = 1]  cluster_init_ = -np.ones(Nele, dtype = int)
    Nclus = len_centers - len_to_remove

    for i in range(len_centers - len_to_remove):
        cluster_init_[centers[i]] = i

    if verb:
        print("Number of clusters before multimodality test=", Nclus)

    sortg = np.argsort(-g)  # Rank of the elements in the g vector sorted in descendent order

    # Perform preliminar assignation to clusters
    maxk = dist_indices.shape[1]
    print(maxk)
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


    sec2 = time.time()
    if verb: print(
         "{0:0.2f} seconds clustering before multimodality test".format(sec2 - sec))






    cdef np.ndarray[floatTYPE_t, ndim = 2]  Rho_bord = np.zeros((Nclus, Nclus))
    cdef np.ndarray[floatTYPE_t, ndim = 2]  Rho_bord_err = np.zeros((Nclus, Nclus))
    cdef np.ndarray[DTYPE_t, ndim = 2]  Point_bord = np.zeros((Nclus, Nclus), dtype=int)

    cdef np.ndarray[floatTYPE_t, ndim = 1]  pos = np.zeros(Nclus * Nclus)
    cdef np.ndarray[DTYPE_t, ndim = 1]  ipos = np.zeros(Nclus * Nclus, dtype=int)
    cdef np.ndarray[DTYPE_t, ndim = 1]  jpos = np.zeros(Nclus * Nclus, dtype=int)

    cdef np.ndarray[DTYPE_t, ndim = 1]  cluster_init = np.array(cluster_init_)


    # Find border points between putative clusters
    sec = time.time()

    saddle_count = 0
    for c in range(Nclus):
        #saddle point index, saddle point density, saddle point error, cluster1 index, cluster2 index, valid_saddle
        #create another array of indices

        saddle_density_tmp = np.zeros((Nclus, 3), dtype = float)    #density, density_error, normalized_density
        saddle_indices_tmp = -np.ones((Nclus, 4), dtype = int)      #saddle point, cluster1, cluster2, is_valid saddle
        #log_den_bord_tmp = -np.ones((Nclus, 6), dtype=float)

        if c == 0:
            saddle_density = saddle_density_tmp
            saddle_indices = saddle_indices_tmp                                 #array of saddle points of cluster  c (will be eventually much less than Nclus)
        else:
            saddle_density = np.concatenate((saddle_density, saddle_density_tmp), axis = 0)
            saddle_indices = np.concatenate((saddle_indices, saddle_indices_tmp), axis = 0)

        for p1 in cl_struct[c]:                                 #p1 point in a given cluster
            if p1 in centers:
                pp = -1

            else:
                #a point p2 in the neighborhood of p1 belongs to a cluster cp different from p1
                for k in range(1, self.kstar[p1] + 1):
                    p2 = self.dist_indices[p1, k]
                    pp = -1
                    if cluster_init[p2] != c:               #a point in the neighborhood of p1 belongs to another cluster
                        pp = p2
                        cp = cluster_init[pp]               #neighbor cluster index
                        break

            if pp != -1:
                #p1 is the closest point in c to p2
                for k in range(1, self.maxk):               #why maxk?
                    po = self.dist_indices[pp, k]           #pp here is p2
                    if po == p1:                            #p1 is the closest point of p2 belonging to c
                        break
                    if cluster_init[po] == c:               #p1 is NOT the closest point of p2 belonging to c
                        pp = -1
                        break

            #here check if the border point is a saddle point
            if pp != -1:
                #maybe contrunct a matrix of border points [p, g, c1, c2]
                #in log_den_bord_tmp the cluster are coupled with indices that go from smaller to bigger
                if c>cp:
                    c1 = cp
                    c2 = c
                else:
                    c1 = c
                    c2 = cp

                flag = 0
                for i in range(saddle_count):
                    if c1 == saddle_indices[i, 1] and c2 == saddle_indices[i, 2]:
                        flag = 1    #there is already a border point between c and cp
                        if g[p1] > saddle_density[i, 2]:
                            #if c <1:                    #check that the which density is bigger
                                #print(g[p1], saddle_density[i, 0], p1, c1, c2)
                            saddle_indices[i, 0] = p1                         #useful at the end
                            saddle_density[i] = np.array( [log_den_c[p1], self.log_den_err[p1], g[p1]] )
                            break

                if flag == 0:
                    #if c <1:
                        #print(g[p1], saddle_density[saddle_count, 0], p1, c1, c2)
                    saddle_indices[saddle_count] = np.array([p1, c1, c2, 1], dtype = int)
                    saddle_density[saddle_count] = np.array( [log_den_c[p1], self.log_den_err[p1], g[p1]] )
                    saddle_count+=1


        saddle_density = saddle_density[:saddle_count]
        saddle_indices = saddle_indices[:saddle_count]

    sort_indices = saddle_density[:, 0].argsort()[::-1]
    saddle_density = saddle_density[sort_indices]
    saddle_indices = saddle_indices[sort_indices]

    sec2 = time.time()
    if verb:
        print("{0:0.2f} seconds identifying the borders".format(sec2 - sec))








    cdef np.ndarray[DTYPE_t, ndim = 1]  centers_ = np.array(centers, dtype=int)
    cdef np.ndarray[DTYPE_t, ndim = 1]  clsurv = np.ones(Nclus, dtype=int)

    sec = time.time()
    # Here we start the merging process through multimodality test.
    check = 1
    surviving_clusters = np.ones(Nclus, dtype = int)
    nsaddles = len(saddle_density)

    while check == 1:
        check = 0
        current_saddle = -1.
        for i in range(nsaddles):                                               #log_den_bord already sorted
            if saddle_indices[i, 3]==1:
                clus1, clus2 = saddle_indices[i, 1], saddle_indices[i, 2]      #cluster indices
                center1, center2 = centers[clus1], centers[clus2]              #data indices peak of clus1, clus2

                a1 = log_den_c[center1] - saddle_density[i, 0]      #log_den_c is an array of the density of all the points
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
            saddle_indices[to_remove, -1] = 0             #the couple center1, center2 is removed
            margin1 = max_a1/max_sum_err1
            margin2 = max_a2/max_sum_err2

            #only the peak with the highest margin is kept:
            #by convention we set this peak to be center1
            if margin1 < margin2:
                c1, c2 = saddle_indices[to_remove, 2], saddle_indices[to_remove, 1]
            else:
                c1, c2 = saddle_indices[to_remove, 1], saddle_indices[to_remove, 2]

            surviving_clusters[c2] = 0                  #the second peak is removed
            cl_struct[c1].extend(cl_struct[c2])
            cl_struct[c2] = []

            #select the clusters that are neighbors to cluster1 and cluster2
            #cluster label, saddle label, position of c1/c2 in saddle_indices[:, 1:3]
            neighbors1 = np.zeros((len(saddle_density), 3), dtype = int)
            neighbors2 = np.zeros((len(saddle_density), 3), dtype = int)
            m, l = 0, 0
            for i in range(nsaddles):
                if i != to_remove and saddle_indices[i, 3]==1:
                    cluster_couple = saddle_indices[i, 1:3]

                    if c1 in cluster_couple:
                        neighbors1[l] = np.array([cluster_couple[0], i, 2])
                        if cluster_couple[0] == c1:
                            neighbors1[l] = np.array([cluster_couple[1], i, 1])
                        l+=1

                    elif c2 in cluster_couple:
                        neighbors2[m] = np.array([cluster_couple[0], i, 2])
                        if cluster_couple[0] == c2:
                            neighbors2[m] = np.array([cluster_couple[1], i, 1])
                        m+=1

            neighbors1, neighbors2 = neighbors1[:l], neighbors2[:m]

            #check which are the common neighbors between cluster1 and cluster2
            neighbors1 = neighbors1[neighbors1[:, 0].argsort()]
            neighbors2 = neighbors2[neighbors2[:, 0].argsort()]
            i, j = 0, 0
            while i < len(neighbors1) and j < len(neighbors2):
                if neighbors1[i, 0] == neighbors2[j, 0]:                            #there is a common element
                    index1, index2 = neighbors1[i, 1], neighbors2[j, 1]             #index common neighbor w.r.t cluster1
                    if saddle_density[index1, 0] < saddle_density[index2, 0]:
                        #if the density of the saddle is higher between cluster2 and common neighbor,
                        #use this as new saddle between cluster1 and common neighbor
                        saddle_density[index1, 0] = saddle_density[index2, 0]       #saddle density
                        saddle_density[index1, 1] = saddle_density[index2, 1]       #saddle error
                    #the couples of common elements with the c2 are "deleted"
                    saddle_indices[index2, -1] = 0
                    i += 1
                    j += 1

                elif neighbors1[i, 0] < neighbors2[j, 0]:
                    i += 1
                else:
                    #neighbors of c2 which were not neighbors of c1 become neighbors of c1
                    index2, c2_index = neighbors2[j, 1], neighbors2[j, 2]
                    saddle_indices[index2, c2_index] = c1
                    j += 1

            #some js in the comparison may not have been taken into account
            while j<len(neighbors2):
                index2, c2_index = neighbors2[j, 1], neighbors2[j, 2]
                saddle_indices[index2, c2_index] = c1
                j+=1



    if verb:
        print("{0:0.2f} seconds with multimodality test".format(sec2 - sec))
    sec = time.time()


    #finalize clustering
    N_clusters = 0
    mapping = -np.ones(Nclus, dtype = int)  #all original centers
    cluster_indices, cluster_centers = [], []
    for i in range(Nclus):
        if surviving_clusters[i]==1:
            mapping[i] = N_clusters             #centeri is surviving
            N_clusters+=1
            #!!!!!! check the index i is correct?
            cluster_indices.append(cl_struct[i])
            cluster_centers.append(centers[i])

    bord_indices_m = np.zeros((N_clusters, N_clusters), dtype=int)
    log_den_bord_m = np.zeros((N_clusters, N_clusters), dtype=float)
    log_den_bord_err_m = np.zeros((N_clusters, N_clusters), dtype=float)

    for i in range(len(saddle_density)):   #all original saddles
        if saddle_indices[i, 3]==1:     #a valid saddle is between two valid centers
            j, k = mapping[ saddle_indices[i, 1]  ], mapping[ saddle_indices[i, 2] ]  #map the two valid centers in the corresponding valid new cluster labels
            log_den_bord_m[j, k] = saddle_density[i, 0]
            log_den_bord_err_m[j, k] = saddle_density[i, 1]
            bord_indices_m[j, k] = saddle_indices[i, 0]

    log_den_bord_m += log_den_bord_m.T
    log_den_bord_err_m += log_den_bord_err_m.T
    bord_indices_m += bord_indices_m.T
    log_den_bord_m[np.diag_indices(N_clusters)] =-1

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
