import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import scipy as sp
from scipy.cluster import hierarchy
from sklearn import manifold


def plot_ID_line_fit_estimation(Data, decimation = 0.9, fraction_used=0.9):

    mus = Data.distances[:, 2] / Data.distances[:, 1]

    idx  = np.arange(mus.shape[0])
    idx = np.random.choice(idx, size=(int(np.around(Data.Nele*decimation))), replace = False)
    mus = mus[idx]

    mus = np.sort(np.sort(mus))

    Nele_eff = int(np.around(fraction_used * Data.Nele, decimals=0))

    x = np.log(mus)
    y = -np.log(1. - np.arange(0, mus.shape[0]) / mus.shape[0])

    x_, y_ = np.atleast_2d(x[:Nele_eff]).T, y[:Nele_eff]

    slope, residuals, rank, s = np.linalg.lstsq(x_, y_, rcond=None)#x[:Nele_eff, None]?

    plt.plot(x, y, 'o')
    plt.plot(x[:Nele_eff], y[:Nele_eff], 'o')
    plt.plot(x, x * slope, '-')

    print('slope is {:f}, average resitual is {:f}'.format(slope[0],
                                                           residuals[0] / Nele_eff))

    plt.xlabel('log(mu)')
    plt.ylabel('-log(1-F(mu))')
    #plt.savefig('ID_line_fit_plot.png')


def plot_ID_vs_fraction(Data, fractions = np.linspace(0.1, 1, 25)):

    IDs = []
    IDs_errs = []
    verbose = Data.verb
    Data.verb = False

    for frac in fractions:
        Data.compute_id(decimation=frac, n_reps=5)
        IDs.append(Data.id_estimated_ml)
        IDs_errs.append(Data.id_estimated_ml_std)

    Data.verb = verbose
    plt.errorbar(fractions*Data.Nele, IDs, yerr=IDs_errs)
    plt.xlabel('N')
    plt.ylabel('estimated ID')
    #plt.savefig('ID_decimation_plot.png')


def plot_ID_vs_nneigh(Data, nneighs = np.arange(2, 90)):

    IDs = []
    IDs_errs = []
    verbose = Data.verb
    Data.verb = False

    for nneigh in nneighs:
        Data.compute_id_diego(nneigh=nneigh)
        IDs.append(Data.id_estimated_ml)


    Data.verb = verbose
    plt.plot(1./nneighs, IDs)
    #plt.xscale('log')
    plt.xlabel('1/nneigh')
    plt.ylabel('estimated ID')
    #plt.savefig('ID_neighs_plot.png')


def plot_SLAn(Data, linkage = 'single'):
    assert (Data.labels is not None)

    nd = int((Data.Nclus_m * Data.Nclus_m - Data.Nclus_m) / 2)
    Dis = np.empty(nd, dtype=float)
    nl = 0
    Fmax = max(Data.Rho)
    Rho_bord_m = np.copy(Data.out_bord)

    for i in range(Data.Nclus_m - 1):
        for j in range(i + 1, Data.Nclus_m):
            Dis[nl] = Fmax - Rho_bord_m[i][j]
            nl = nl + 1
    
    if linkage == 'single':
        DD = sp.cluster.hierarchy.single(Dis)
    elif linkage == 'complete':
        DD = sp.cluster.hierarchy.complete(Dis)
    elif linkage == 'average':
        DD = sp.cluster.hierarchy.average(Dis)
    elif linkage == 'weighted':
        DD = sp.cluster.hierarchy.weighted(Dis)
    else:
        print('ERROR: select a valid linkage criterion')
        
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    dn = sp.cluster.hierarchy.dendrogram(DD)
    #fig.savefig('dendrogramm.png')  # save the figure to file
    plt.show()
    # plt.close(fig)  # close the figure


def plot_MDS(Data):
    Fmax = max(Data.Rho)
    Rho_bord_m = np.copy(Data.out_bord)
    d_dis = np.zeros((Data.Nclus_m, Data.Nclus_m), dtype=float)
    model = manifold.MDS(n_components=2, n_jobs=None, dissimilarity='precomputed')
    for i in range(Data.Nclus_m):
        for j in range(Data.Nclus_m):
            d_dis[i][j] = Fmax - Rho_bord_m[i][j]
    for i in range(Data.Nclus_m):
        d_dis[i][i] = 0.
    out = model.fit_transform(d_dis)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    s = []
    col = []
    for i in range(Data.Nclus_m):
        s.append(20. * np.sqrt(len(Data.clstruct_m[i])))
        col.append(i)
    plt.scatter(out[:, 0], out[:, 1], s=s, c=col)
    for i in range(Data.Nclus_m):
        ax.annotate(i, (out[i, 0], out[i, 1]))
    # Add edges
    rr = np.amax(Rho_bord_m)
    if (rr > 0.):
        Rho_bord_m = Rho_bord_m / rr * 100.
    start_idx, end_idx = np.where(out)
    segments = [[out[i, :], out[j, :]]
                for i in range(len(out)) for j in range(len(out))]
    values = np.abs(Rho_bord_m)
    lc = LineCollection(segments, zorder=0, norm=plt.Normalize(0, values.max()))
    lc.set_array(Rho_bord_m.flatten())
    # lc.set_linewidths(np.full(len(segments),0.5))
    lc.set_edgecolor(np.full(len(segments), 'black'))
    lc.set_facecolor(np.full(len(segments), 'black'))
    lc.set_linewidths(0.02 * Rho_bord_m.flatten())
    ax.add_collection(lc)
    #fig.savefig('2D.png')
    plt.show()
    # plt.close(fig)  # close the figure


def plot_matrix(Data):
    Rho_bord_m = np.copy(Data.out_bord)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    topography = np.copy(Rho_bord_m)
    for j in range(Data.Nclus_m):
        topography[j, j] = Data.Rho[Data.centers_m[j]]
    plt.imshow(topography, cmap='hot', interpolation='nearest')
    #fig.savefig('matrix.png')
    plt.show()
    # plt.close(fig)  # close the figure

def plot_inf_imb_plane(imbalances):

    pass

if __name__ == '__main__':
    # generate some random points in n dimensions

    from adpy import Data

    # X = np.vstack(
    #     (np.random.normal(0, 1, size=(1000, 15)), np.random.normal(5, 1, size=(1000, 15))))

    X = np.genfromtxt('Fig1.dat', dtype='float')

    dist = Data(X)

    dist.compute_distances(maxk=200)

    dist.compute_id()

    dist.compute_density_kNN(k=3)

    dist.compute_density_PAk()

    dist.compute_clustering(Z=1.65, halo=True)

    plot_SLAn(dist)

    plot_MDS(dist)

    plot_matrix(dist)
