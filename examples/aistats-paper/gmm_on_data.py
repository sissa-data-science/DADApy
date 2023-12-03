import numpy as np
from sklearn.mixture import GaussianMixture
import time
from dadapy._utils.utils import _align_arrays

datasets = ["data1.txt", "data_2.txt", "data_3.txt"]
free_energies = ["F1.txt", "F2.txt", "F3.txt"]

for d, f in zip(datasets, free_energies):
    X = np.genfromtxt("datasets/{}".format(d))
    F_true = np.genfromtxt("datasets/{}".format(f))

    Nsample = X.shape[0]

    sec = time.perf_counter()
    maxn_components = min(int(Nsample/2)+1, 10)

    gmms = np.array([GaussianMixture(n_components=n_components).fit(X).bic(X) for n_components in range(1, maxn_components)])
    best_n_components = np.argmin(gmms) + 1
    gmm = GaussianMixture(n_components=best_n_components)
    gmm.fit(X)
    log_dens = gmm.score_samples(X)
    F_predicted = -log_dens
    time_GMM = time.perf_counter() - sec

    _, F_predicted = _align_arrays(F_predicted, np.ones_like(F_true), F_true)
    MAE_GMM = np.mean(np.abs(F_predicted - F_true))
    MSE_GMM = np.mean((F_predicted - F_true) ** 2)
