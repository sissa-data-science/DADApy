from sklearn.mixture import GaussianMixture
import numpy as np
from utils_rebuttal import den_6d
from dadapy._utils.utils import _align_arrays


# import 6 dimensional data from datasets folder
data = np.genfromtxt("datasets/6d_double_well-1.2M-last_400k.txt", dtype="float32")
X = data[-40000:, :]

#
gmms = np.array([GaussianMixture(n_components=n_components).fit(X).bic(X) for n_components in range(1, 10)])

best_n_components = np.argmin(gmms) + 1

gmm = GaussianMixture(n_components=best_n_components)
gmm.fit(X)

log_dens = gmm.score_samples(X)
F_predicted = -log_dens

# compare to true density
true_dens = np.array([np.log(den_6d(x)) for x in X])
F_true = -true_dens

# align
_, F_predicted = _align_arrays(F_predicted, np.ones_like(F_true), F_true)

# compute MAE and MSE
MAE = np.mean(np.abs(F_predicted - F_true))
MSE = np.mean((F_predicted - F_true) ** 2)
print("MAE: ", MAE)
print("MSE: ", MSE)


# plot
import matplotlib.pyplot as plt

plt.scatter(F_true, F_predicted, s=1)
plt.show()