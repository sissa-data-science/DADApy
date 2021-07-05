import multiprocessing
import time
import numpy as np
import scipy as sp

from duly.cython_ import cython_clustering as cf
from duly.density_estimation import DensityEstimation

cores = multiprocessing.cpu_count()


class Clustering(DensityEstimation):
    """This class contains various density-based clustering algorithms.

    Inherits from the DensityEstimation class.

    Attributes:

        Nclus_m (int) : number of clusters found
        labels (list(int)): cluster assignment. A list of length Nele containg the cluster assignment of each point
        as an integer from 0 to Nclus_m-1.
        centers_m (int): Indices of the centroids of each cluster (density peak)
        clstruct_m (list(list(int))): a list of lists. Each sublist contains the indices belonging to the
        corresponding cluster.
        out_bord (array(float)): an array of dimensions Nclus_m x Nclus_m containg the estimated log density of the
        the saddle point between each couple of peaks.
        Rho_bord_err_m (array(float)): an array of dimensions Nclus_m x Nclus_m containg the estimated error on the
        log density of the saddle point between each couple of peaks.

    """

    def __init__(
        self, coordinates=None, distances=None, maxk=None, verbose=False, njobs=cores
    ):
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            verbose=verbose,
            njobs=njobs,
        )

        self.clstruct_m = None
        self.Nclus_m = None
        self.labels = None
        self.centers_m = None
        self.Rho_bord_err_m = None
        self.out_bord = None

        self.delta= None # Minimum distance from an element with higher density
        self.ref= None   # Index of the nearest element with higher density

    def compute_clustering_optimised(self, Z=1.65, halo=False):
        assert self.Rho is not None
        if self.verb:
            print("Clustering started")

        # Make all values of Rho positives (this is important to help convergence)
        Rho_min = np.min(self.Rho)

        Rho_c = self.Rho + np.log(self.Nele)
        Rho_c = Rho_c - Rho_min + 1

        # Putative modes of the PDF as preliminary clusters

        Nele = self.distances.shape[0]
        g = Rho_c - self.Rho_err
        # centers are point of max density  (max(g) ) within their optimal neighborhood (defined by kstar)
        seci = time.time()

        out = cf._compute_clustering(
            Z,
            halo,
            self.kstar,
            self.dist_indices.astype(int),
            self.maxk,
            self.verb,
            self.Rho_err,
            Rho_min,
            Rho_c,
            g,
            Nele,
        )

        secf = time.time()

        self.clstruct_m = out[0]
        self.Nclus_m = out[1]
        self.labels = out[2]
        self.centers_m = out[3]
        out_bord = out[4]
        Rho_min = out[5]
        self.Rho_bord_err_m = out[6]

        self.out_bord = out_bord + Rho_min - 1 - np.log(Nele)

        if self.verb:
            print("Clustering finished, {} clusters found".format(self.Nclus_m))
            print("total time is, {}".format(secf - seci))

    def compute_DecGraph(self):
        assert self.Rho is not None
        assert self.X is not None
        self.delta=np.zeros(self.Nele)
        self.ref=np.zeros(self.Nele,dtype='int')
        tt=np.arange(self.Nele)
        imax=[]
        ncalls=0
        for i in range(self.Nele):
            ll=tt[((self.Rho>self.Rho[i]) & (tt!=i))]
            if (ll.shape[0]>0):
                a1,a2,a3=np.intersect1d(self.dist_indices[i,:],ll,return_indices=True,assume_unique=True)
                if (a1.shape[0]>0):
                    aa=np.min(a2)
                    self.delta[i]=self.distances[i,aa]
                    self.ref[i]=self.dist_indices[i,aa]
                else:
                    ncalls=ncalls+1
                    dd=self.X[((self.Rho>self.Rho[i]) & (tt!=i))]
                    ds=np.transpose(sp.spatial.distance.cdist([np.transpose(self.X[i,:])],dd))
                    j=np.argmin(ds)
                    self.ref[i]=ll[j]
                    self.delta[i]=ds[j]
            else:
                self.delta[i]=-100.
                imax.append(i)
        self.delta[imax]=1.05*np.max(self.delta)
        print ("Number of points for which self.delta needed call to cdist=",ncalls)
    def compute_cluster_DP(self, dens_cut=0.,delta_cut=0., halo=False):
        assert self.delta is not None
        ordered=np.argsort(-self.Rho)
        self.labels=np.zeros(self.Nele,dtype='int')
        tt=np.arange(self.Nele)
        center_label=np.zeros(self.Nele,dtype='int')
        ncluster=-1
        for i in range(self.Nele):
            j=ordered[i]
            if ((self.Rho[j]>dens_cut) & (self.delta[j] >delta_cut)):
                ncluster=ncluster+1
                self.labels[j]=ncluster
                center_label[j]=ncluster
            else:
                self.labels[j]=self.labels[self.ref[j]]
                center_label[j]=-1
        self.centers=tt[(center_label!=-1)]
        if (halo):
            bord=np.zeros(self.Nele,dtype='int')
            halo=np.copy(self.labels)

            for i in range(self.Nele):
                for j in self.dist_indices[i,:][(self.distances[i,:]<=self.dc[i])]:
                    if (self.labels[i]!=self.labels[j]):
                        bord[i]=1
            halo_cutoff=np.zeros(ncluster+1)
            halo_cutoff[:]=np.min(self.Rho)-1
            for j in range (ncluster+1):
                td=self.Rho[((bord==1)&(self.labels==j))]
                if (td.size != 0):
                    halo_cutoff[j]=np.max(td)
            halo[tt[(self.Rho<halo_cutoff[self.labels])]]=-1
            self.labels=halo



    def compute_clustering(self, Z=1.65, halo=False):
        assert self.Rho is not None
        if self.verb:
            print("Clustering started")

        # Make all values of Rho positives (this is important to help convergence)
        Rho_min = np.min(self.Rho)
        Rho_c = self.Rho + np.log(self.Nele)
        Rho_c = Rho_c - Rho_min + 1

        # Putative modes of the PDF as preliminary clusters
        sec = time.time()
        Nele = self.distances.shape[0]
        g = Rho_c - self.Rho_err
        centers = []
        for i in range(Nele):
            t = 0
            for j in range(1, self.kstar[i] + 1):
                if g[i] < g[self.dist_indices[i, j]]:
                    t = 1
                    break
            if t == 0:
                centers.append(i)
        for i in centers:
            l, m = np.where(self.dist_indices == i)
            for j in range(l.shape[0]):
                if (g[l[j]] > g[i]) & (m[j] <= self.kstar[l[j]]):
                    centers.remove(i)
                    break
        cluster_init = []
        for j in range(Nele):
            cluster_init.append(-1)
            Nclus = len(centers)
        if self.verb:
            print("Number of clusters before multimodality test=", Nclus)

        for i in centers:
            cluster_init[i] = centers.index(i)
        sortg = np.argsort(-g)  # Get the rank of the elements in the g vector
        # sorted in descendent order.
        # Perform preliminar assignation to clusters
        for j in range(Nele):
            ele = sortg[j]
            nn = 0
            while cluster_init[ele] == -1:
                nn = nn + 1
                cluster_init[ele] = cluster_init[self.dist_indices[ele, nn]]
        clstruct = []  # useful list of points in the clusters
        for i in range(Nclus):
            x1 = []
            for j in range(Nele):
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
        Rho_bord = np.zeros((Nclus, Nclus), dtype=float)
        Rho_bord_err = np.zeros((Nclus, Nclus), dtype=float)
        Point_bord = np.zeros((Nclus, Nclus), dtype=int)
        # Find border points between putative clusters
        sec = time.time()
        for i in range(Nclus):
            for j in range(Nclus):
                Point_bord[i][j] = -1
        for c in range(Nclus):
            for p1 in clstruct[c]:
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
                    if g[p1] > Rho_bord[c][cp]:
                        Rho_bord[c][cp] = g[p1]
                        Rho_bord[cp][c] = g[p1]
                        Point_bord[cp][c] = p1
                        Point_bord[c][cp] = p1
                    # if (g[pp]>Rho_bord[c][cp]):
                    #     Rho_bord[c][cp]=g[pp]
                    #     Rho_bord[cp][c]=g[pp]
                    #     Point_bord[cp][c]=pp
                    #     Point_bord[c][cp]=pp
        for i in range(Nclus - 1):
            for j in range(i + 1, Nclus):
                if Point_bord[i][j] != -1:
                    Rho_bord[i][j] = Rho_c[Point_bord[i][j]]
                    Rho_bord[j][i] = Rho_c[Point_bord[j][i]]
                    Rho_bord_err[i][j] = self.Rho_err[Point_bord[i][j]]
                    Rho_bord_err[j][i] = self.Rho_err[Point_bord[j][i]]
        for i in range(Nclus):
            Rho_bord[i][i] = -1.0
            Rho_bord_err[i][i] = 0.0
        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds identifying the borders".format(sec2 - sec))
        check = 1
        clsurv = []
        sec = time.time()
        for i in range(Nclus):
            clsurv.append(1)
        while check == 1:
            pos = []
            ipos = []
            jpos = []
            check = 0
            for i in range(Nclus - 1):
                for j in range(i + 1, Nclus):
                    a1 = Rho_c[centers[i]] - Rho_bord[i][j]
                    a2 = Rho_c[centers[j]] - Rho_bord[i][j]
                    e1 = Z * (self.Rho_err[centers[i]] + Rho_bord_err[i][j])
                    e2 = Z * (self.Rho_err[centers[j]] + Rho_bord_err[i][j])
                    if a1 < e1 or a2 < e2:
                        check = 1
                        pos.append(Rho_bord[i][j])
                        ipos.append(i)
                        jpos.append(j)
            if check == 1:
                barriers = pos.index(max(pos))
                imod = ipos[barriers]
                jmod = jpos[barriers]
                if Rho_c[centers[imod]] < Rho_c[centers[jmod]]:
                    tmp = jmod
                    jmod = imod
                    imod = tmp
                clsurv[jmod] = 0
                Rho_bord[imod][jmod] = -1.0
                Rho_bord[jmod][imod] = -1.0
                Rho_bord_err[imod][jmod] = 0.0
                Rho_bord_err[jmod][imod] = 0.0
                clstruct[imod].extend(clstruct[jmod])
                clstruct[jmod] = []
                for i in range(Nclus):
                    if i != imod and i != jmod:
                        if Rho_bord[imod][i] < Rho_bord[jmod][i]:
                            Rho_bord[imod][i] = Rho_bord[jmod][i]
                            Rho_bord[i][imod] = Rho_bord[imod][i]
                            Rho_bord_err[imod][i] = Rho_bord_err[jmod][i]
                            Rho_bord_err[i][imod] = Rho_bord_err[imod][i]
                        Rho_bord[jmod][i] = -1
                        Rho_bord[i][jmod] = Rho_bord[jmod][i]
                        Rho_bord_err[jmod][i] = 0
                        Rho_bord_err[i][jmod] = Rho_bord_err[jmod][i]
        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds with multimodality test".format(sec2 - sec))
        Nclus_m = 0
        clstruct_m = []
        centers_m = []
        nnum = []
        for j in range(Nclus):
            nnum.append(-1)
            if clsurv[j] == 1:
                nnum[j] = Nclus_m
                Nclus_m = Nclus_m + 1
                clstruct_m.append(clstruct[j])
                centers_m.append(centers[j])
        Rho_bord_m = np.zeros((Nclus_m, Nclus_m), dtype=float)
        Rho_bord_err_m = np.zeros((Nclus_m, Nclus_m), dtype=float)
        for j in range(Nclus):
            if clsurv[j] == 1:
                jj = nnum[j]
                for k in range(Nclus):
                    if clsurv[k] == 1:
                        kk = nnum[k]
                        Rho_bord_m[jj][kk] = Rho_bord[j][k]
                        Rho_bord_err_m[jj][kk] = Rho_bord_err[j][k]
        Last_cls = np.empty(Nele, dtype=int)
        for j in range(Nclus_m):
            for k in clstruct_m[j]:
                Last_cls[k] = j
        Last_cls_halo = np.copy(Last_cls)
        nh = 0
        for j in range(Nclus_m):
            Rho_halo = max(Rho_bord_m[j])
            for k in clstruct_m[j]:
                if Rho_c[k] < Rho_halo:
                    nh = nh + 1
                    Last_cls_halo[k] = -1
        if halo:
            labels = Last_cls_halo
        else:
            labels = Last_cls
        out_bord = np.copy(Rho_bord_m)
        self.clstruct_m = clstruct_m
        self.Nclus_m = Nclus_m
        self.labels = labels
        self.centers_m = centers_m
        self.out_bord = (
            out_bord + Rho_min - 1 - np.log(Nele)
        )  # remove wrong normalisation introduced earlier
        self.Rho_bord_err_m = Rho_bord_err_m
        if self.verb:
            print("Clustering finished, {} clusters found".format(self.Nclus_m))


# if __name__ == '__main__':
#     X = np.random.uniform(size=(50, 2))
#
#     cl = Clustering(coordinates=X)
#
#     cl.compute_distances(maxk=25)
#
#     cl.compute_id_2NN()
#
#     cl.compute_density_kNN(10)
#
#     cl.compute_clustering()
#
#     print(cl.Nclus_m)
