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

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib import cm
from matplotlib.collections import LineCollection
from scipy import cluster
from sklearn import manifold


def plot_ID_line_fit_estimation(Data, decimation=0.9, fraction_used=0.9):
    mus = Data.distances[:, 2] / Data.distances[:, 1]

    idx = np.arange(mus.shape[0])
    idx = np.random.choice(
        idx, size=(int(np.around(Data.N * decimation))), replace=False
    )
    mus = mus[idx]

    mus = np.sort(np.sort(mus))

    Nele_eff = int(np.around(fraction_used * Data.N, decimals=0))

    x = np.log(mus)
    y = -np.log(1.0 - np.arange(0, mus.shape[0]) / mus.shape[0])

    x_, y_ = np.atleast_2d(x[:Nele_eff]).T, y[:Nele_eff]

    slope, residuals, rank, s = np.linalg.lstsq(
        x_, y_, rcond=None
    )  # x[:Nele_eff, None]?

    plt.plot(x, y, "o")
    plt.plot(x[:Nele_eff], y[:Nele_eff], "o")
    plt.plot(x, x * slope, "-")

    print(
        "slope is {:f}, average resitual is {:f}".format(
            slope[0], residuals[0] / Nele_eff
        )
    )

    plt.xlabel("log(mu)")
    plt.ylabel("-log(1-F(mu))")
    # plt.savefig('ID_line_fit_plot.png')


def plot_ID_vs_fraction(Data, fractions=np.linspace(0.1, 1, 25)):
    IDs = []
    IDs_errs = []
    verbose = Data.verb
    Data.verb = False

    for frac in fractions:
        Data.compute_id(decimation=frac, n_reps=5)
        IDs.append(Data.id_estimated_ml)
        IDs_errs.append(Data.id_estimated_ml_std)

    Data.verb = verbose
    plt.errorbar(fractions * Data.N, IDs, yerr=IDs_errs)
    plt.xlabel("N")
    plt.ylabel("estimated ID")
    # plt.savefig('ID_decimation_plot.png')


def plot_ID_vs_nneigh(Data, nneighs=np.arange(2, 90)):
    IDs = []
    verbose = Data.verb
    Data.verb = False

    for nneigh in nneighs:
        Data.compute_id_diego(nneigh=nneigh)
        IDs.append(Data.id_estimated_ml)

    Data.verb = verbose
    plt.plot(1.0 / nneighs, IDs)
    # plt.xscale('log')
    plt.xlabel("1/nneigh")
    plt.ylabel("estimated ID")
    # plt.savefig('ID_neighs_plot.png')


def plot_SLAn(Data, linkage="single"):
    assert Data.cluster_assignment is not None

    nd = int((Data.N_clusters * Data.N_clusters - Data.N_clusters) / 2)
    Dis = np.empty(nd, dtype=float)
    nl = 0
    Fmax = max(Data.log_den)
    Rho_bord_m = np.copy(Data.log_den_bord)

    for i in range(Data.N_clusters - 1):
        for j in range(i + 1, Data.N_clusters):
            Dis[nl] = Fmax - Rho_bord_m[i][j]
            nl = nl + 1

    if linkage == "single":
        DD = sp.cluster.hierarchy.single(Dis)
    elif linkage == "complete":
        DD = sp.cluster.hierarchy.complete(Dis)
    elif linkage == "average":
        DD = sp.cluster.hierarchy.average(Dis)
    elif linkage == "weighted":
        DD = sp.cluster.hierarchy.weighted(Dis)
    else:
        print("ERROR: select a valid linkage criterion")

    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    dn = sp.cluster.hierarchy.dendrogram(DD)
    # fig.savefig('dendrogramm.png')  # save the figure to file
    plt.show()
    # plt.close(fig)  # close the figure


def plot_MDS(Data, cmap="viridis"):
    Fmax = max(Data.log_den)
    Rho_bord_m = np.copy(Data.log_den_bord)
    d_dis = np.zeros((Data.N_clusters, Data.N_clusters), dtype=float)
    model = manifold.MDS(n_components=2, n_jobs=None, dissimilarity="precomputed")
    for i in range(Data.N_clusters):
        for j in range(Data.N_clusters):
            d_dis[i][j] = Fmax - Rho_bord_m[i][j]
    for i in range(Data.N_clusters):
        d_dis[i][i] = 0.0
    out = model.fit_transform(d_dis)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    s = []
    col = []
    for i in range(Data.N_clusters):
        s.append(20.0 * np.sqrt(len(Data.cluster_indices[i])))
        col.append(i)
    plt.scatter(out[:, 0], out[:, 1], s=s, c=col, cmap=cmap)
    cmal = cm.get_cmap(cmap, Data.N_clusters)
    colors = cmal(np.arange(0, cmal.N))
    for i in range(Data.N_clusters):
        cc = "k"
        r = colors[i][0]
        g = colors[i][1]
        b = colors[i][2]
        luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) * 255
        if luma < 156:
            cc = "w"
        plt.annotate(
            i,
            (out[i, 0], out[i, 1]),
            horizontalalignment="center",
            verticalalignment="center",
            c=cc,
            weight="bold",
        )
    #    for i in range(Data.N_clusters):
    #        ax.annotate(i, (out[i, 0], out[i, 1]))
    # Add edges
    rr = np.amax(Rho_bord_m)
    if rr > 0.0:
        Rho_bord_m = Rho_bord_m / rr * 100.0
    start_idx, end_idx = np.where(out)
    segments = [
        [out[i, :], out[j, :]] for i in range(len(out)) for j in range(len(out))
    ]
    values = np.abs(Rho_bord_m)
    lc = LineCollection(segments, zorder=0, norm=plt.Normalize(0, values.max()))
    lc.set_array(Rho_bord_m.flatten())
    # lc.set_linewidths(np.full(len(segments),0.5))
    lc.set_edgecolor(np.full(len(segments), "black"))
    lc.set_facecolor(np.full(len(segments), "black"))
    lc.set_linewidths(0.02 * Rho_bord_m.flatten())
    ax.add_collection(lc)
    # fig.savefig('2D.png')
    plt.show()
    # plt.close(fig)  # close the figure


def plot_matrix(Data):
    Rho_bord_m = np.copy(Data.log_den_bord)
    topography = np.copy(Rho_bord_m)
    for j in range(Data.N_clusters):
        topography[j, j] = Data.log_den[Data.cluster_centers[j]]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.imshow(topography, cmap="hot", interpolation=None)
    # fig.savefig('matrix.png')
    plt.show()
    # plt.close(fig)  # close the figure


def plot_DecGraph(Data):
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$\delta$")
    plt.scatter(Data.log_den, Data.delta)
    plt.show()


def get_dendrogram(Data, cmap="viridis", savefig="", logscale=True):
    #
    # get_dendrogram (Data,cmap="viridis",savefig"",logscale=True)
    #
    # This function generates a visualization of the topography computed with
    # ADP. Fundamentaly it corresponds to a hierarchy of the clusters build
    # with Single Linkage taking as similarity measure the density at the
    # border between clusters. At difference from classical dendrograms,
    # where all the branches have the same height, in this case
    # the height of the branches is proportional to the density of the cluster
    # centre. To convey more information, the distance in the x-axis between
    # clusters is proportional to the population (or its logarithm).
    # It takes as mandatory argument:
    #
    # Data: A dadapy data object for which ADP has been already run.
    #
    # While the optional arguments are:
    #
    # cmap: The color map for representing the different clusters,
    # the default is "viridis".
    # savefig: A string with the name of the file in which the dendrogram
    # will be saved. The default is empty, so no file is generated.
    # logscale: Makes the distances in the x-axis between clusters proportional
    # to the logarithm of the population of the clusters instead of
    # proportional to the population itself. In very unbalanced clusterings,
    # it makes the dendrogram more human readable. The default is True.
    #
    # Generation of SL dendrogram
    # Prepare some auxiliary lists
    e1 = []
    e2 = []
    d12 = []
    L = []
    Li1 = []
    Li2 = []
    Ldis = []
    Fmax = max(Data.log_den)
    Rho_bord_m = np.copy(Data.log_den_bord)
    # Obtain populations of the clusters for fine tunning the x-axis
    pop = np.zeros((Data.N_clusters), dtype=int)
    for i in range(Data.N_clusters):
        pop[i] = len(Data.cluster_indices[i])
        if logscale:
            pop[i] = np.log(pop[i])
    xr = np.sum(pop)
    # Obtain distances in list format from topography
    for i in range(Data.N_clusters - 1):
        for j in range(i + 1, Data.N_clusters):
            dis12 = Fmax - Rho_bord_m[i][j]
            e1.append(i)
            e2.append(j)
            d12.append(dis12)

    # Obtain the dendrogram in form of links
    nlinks = 0
    clnew = Data.N_clusters
    for j in range(Data.N_clusters - 1):
        aa = np.argmin(d12)
        nlinks = nlinks + 1
        L.append(clnew + nlinks)
        Li1.append(e1[aa])
        Li2.append(e2[aa])
        Ldis.append(d12[aa])
        # update distance matrix
        t = 0
        fe = Li1[nlinks - 1]
        fs = Li2[nlinks - 1]
        newname = L[nlinks - 1]
        # list of untouched clusters
        unt = []
        for r in d12:
            if (e1[t] != fe) & (e1[t] != fs):
                unt.append(e1[t])
            if (e2[t] != fe) & (e2[t] != fs):
                unt.append(e2[t])
            t = t + 1
        myset = set(unt)
        unt = list(myset)
        # Build a new distance matrix
        e1new = []
        e2new = []
        d12new = []
        for j in unt:
            t = 0
            dmin = 9.9e99
            for r in d12:
                if (e1[t] == j) | (e2[t] == j):
                    if (e1[t] == fe) | (e2[t] == fe) | (e1[t] == fs) | (e2[t] == fs):
                        if d12[t] < dmin:
                            dmin = d12[t]
                t = t + 1
            e1new.append(j)
            e2new.append(newname)
            d12new.append(dmin)

        t = 0
        for r in d12:
            if (unt.count(e1[t])) & (unt.count(e2[t])):
                e1new.append(e1[t])
                e2new.append(e2[t])
                d12new.append(d12[t])
            t = t + 1

        e1 = e1new
        e2 = e2new
        d12 = d12new

    # Get the order in which the elements should be displayed
    sorted_elements = []
    sorted_elements.append(L[nlinks - 1])

    for jj in range(len(L)):
        j = len(L) - jj - 1
        for i in range(len(sorted_elements)):
            if sorted_elements[i] == L[j]:
                sorted_elements[i] = Li2[j]
                sorted_elements.insert(i, Li1[j])

    add = 0.0
    x = []
    y = []
    label = []
    for i in range(len(sorted_elements)):
        label.append(sorted_elements[i])
        j = Data.cluster_centers[label[i]]
        y.append(Data.log_den[j])
        x.append(add + 0.5 * pop[sorted_elements[i]])
        add = add + pop[sorted_elements[i]]

    xs = x.copy()
    ys = y.copy()
    labels = label.copy()
    zorder = 0
    for jj in range(len(L)):
        c1 = label.index(Li1[jj])
        c2 = label.index(Li2[jj])
        label.append(L[jj])
        x.append((x[c1] + x[c2]) / 2.0)
        ynew = Fmax - Ldis[jj]
        y.append(ynew)
        x1 = x[c1]
        y1 = y[c1]
        x2 = x[c2]
        y2 = y[c2]
        zorder = zorder + 1
        plt.plot(
            [x1, x1], [y1, ynew], color="k", linestyle="-", linewidth=2, zorder=zorder
        )
        zorder = zorder + 1
        plt.plot(
            [x2, x2], [y2, ynew], color="k", linestyle="-", linewidth=2, zorder=zorder
        )
        zorder = zorder + 1
        plt.plot(
            [x1, x2], [ynew, ynew], color="k", linestyle="-", linewidth=2, zorder=zorder
        )

    zorder = zorder + 1
    cmal = cm.get_cmap(cmap, Data.N_clusters)
    colors = cmal(np.arange(0, cmal.N))
    plt.scatter(xs, ys, c=labels, s=100, zorder=zorder, cmap=cmap)
    for i in range(Data.N_clusters):
        zorder = zorder + 1
        cc = "k"
        r = colors[labels[i]][0]
        g = colors[labels[i]][1]
        b = colors[labels[i]][2]
        luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) * 255
        if luma < 156:
            cc = "w"
        plt.annotate(
            labels[i],
            (xs[i], ys[i]),
            horizontalalignment="center",
            verticalalignment="center",
            zorder=zorder,
            c=cc,
            weight="bold",
        )
    plt.xlim([0, xr])
    if savefig != "":
        plt.savefig(savefig)
    plt.show()


def plot_inf_imb_plane(imbalances, coord_list=None, labels=None):
    """Plot the information imbalance plane corresponding to the computed ibalances.

    Args:
        imbalances (np.ndarray): Information imbalances from the full space to specific sets of coordinates and vice-versa
        coord_list (list of lists of integers, optional): The list of coordinates considered for the information
        imbalance computations
        labels (list of strings, optional): Labels for the list of coordinates

    Returns:

    """

    plt.figure(figsize=(4, 4))
    for i, (imb0, imb1) in enumerate(imbalances.T):
        if coord_list is not None:
            if labels is not None:
                label = [labels[c] for c in coord_list[i]]
            else:
                label = coord_list[i]
        else:
            label = ""

        plt.scatter(imb0, imb1, label=label)

    plt.plot([0, 1], [0, 1], "k--")

    if coord_list is not None:
        plt.legend()

    plt.xlabel(r"$\Delta(X_{full} \rightarrow X_{coords}) $")
    plt.ylabel(r"$\Delta(X_{coords} \rightarrow X_{full}) $")


if __name__ == "__main__":
    # generate some random points in n dimensions

    from adpy import Data

    # X = np.vstack(
    #     (np.random.normal(0, 1, size=(1000, 15)),
    #      np.random.normal(5, 1, size=(1000, 15))))

    X = np.genfromtxt("Fig1.dat", dtype="float")

    dist = Data(X)

    dist.compute_distances(maxk=200)

    dist.compute_id()

    dist.compute_density_kNN(k=3)

    dist.compute_density_PAk()

    dist.compute_clustering(Z=1.65, halo=True)

    plot_SLAn(dist)

    get_histogram(dist)

    plot_MDS(dist)

    plot_matrix(dist)
