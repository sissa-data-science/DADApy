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

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()

# from scipy.stats import epps_singleton_2samp as es2s
from scipy.stats import ks_2samp

from ..plot import plot_cdf


def box_counting(
    data,
    box_boundaries,
    input_scales,
    n_offsets=0,
    plot=False,
    verb=True,
):
    """Calculates the fractal dimension of an ensemble of points with given boundaries.
    Be careful, the routine exponentially memory intensive with the number of dimension, we suggest to avoid using it
    if D>10

    Args:
        data (np.ndarray): The data we want to calculate the fractal dimension of.
        box_boundaries (list/tuple/np.ndarray((float,float)): extremes of boxes in each dimension.
            Only square boxes allowed so far.
        input_scales (np.ndarray(float)): array of scales one wants to explore
        n_offsets (int): number of offsets to search over to find the smallest set N(s) to cover all points.
        plot (bool): set to true to see the analytical plot of a calculation.
        verb (bool): when True, print some intermediate information

    Returns:
        scales (np.array(int or float)): size of boxes used to cover the dataset
        ids (np.array(float)): intrinsic dimensions found at different scales
    """

    if data.shape[1] > 10:
        print(
            "The method is memory demanding, you might run out of RAM as the embedding dimension is higher than 10"
        )

    inf = box_boundaries[0]
    sup = box_boundaries[1]

    # count the minimum amount of boxes touched
    Ns = []
    # loop over all scales
    for scale in input_scales:
        bin_temp = np.array(
            [np.arange(inf, sup + 2 * scale, scale) for i in range(data.shape[1])]
        )
        touched = []
        if n_offsets == 0:
            offsets = [0]
        else:
            offsets = np.linspace(0, scale, n_offsets + 1)[:-1]
        # search over all offsets
        for offset in offsets:
            bin_edges = np.array(
                [
                    np.hstack([inf - scale + offset + 1e-5, x + offset + 1e-5])
                    for x in bin_temp
                ]
            )
            H1, e = np.histogramdd(data, bins=bin_edges)
            touched.append(np.sum(H1 > 0))
        Ns.append(touched)
    Ns = np.array(Ns)
    if verb:
        print(Ns)
    # From all sets N found, keep the smallest one at each scale
    Ns = Ns.min(axis=1)
    # Only keep scales at which Ns changed
    scales = np.array([np.min(input_scales[Ns == x]) for x in np.unique(Ns)])
    Ns = np.unique(Ns)
    Ns = Ns[Ns > 0]
    scales = scales[: len(Ns)]
    ind = np.argsort(scales)
    scales = scales[ind]
    Ns = Ns[ind]
    if verb:
        print("effective scales: ", scales, Ns)
    # perform fit with growing number of boxes
    cfs = []
    start = 1
    for i in range(start, len(scales)):
        coeffs = np.polyfit(
            np.log(1 / scales[start - 1 : i + 1]), np.log(Ns[start - 1 : i + 1]), 1
        )
        cfs.append(coeffs[0])

    # print(scales,cfs)
    coeffs = np.polyfit(np.log(1 / scales), np.log(Ns), 1)
    if verb:
        print(np.log(Ns) / np.log(1 / scales[::-1]))
    # make plot
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(np.log(1 / scales), np.log(Ns), c="teal", label="Measured ratios")
        ax.set_ylabel("$\log N(\epsilon)$")
        ax.set_xlabel("$\log 1/ \epsilon$")
        fitted_y_vals = np.polyval(coeffs, np.log(1 / scales))
        ax.plot(
            np.log(1 / scales),
            fitted_y_vals,
            "k--",
            label=f"Fit: {np.round(coeffs[0], 3)}X+{coeffs[1]}",
        )
        ax.legend()
        plt.show()

        plt.figure()
        plt.plot(scales[start:], cfs)
        plt.xlabel("scale")
        plt.ylabel("ID estimated")
        plt.show()

    return np.array(cfs), scales[1:]


# --------------------------------------------------------------------------------------


def correlation_integral(dists, scales, cond=False, plot=True):
    """Calculates the fractal dimension of a D-dimensional ensemble of points using the Correlation Integral
    Compute the intrinsic dimension of the dataset using the so-called Correlation Integral
    (see Grassberger-Procaccia, 1983).

    Args:
        dists (np.ndarray(float,float)): Distances between points.
        scales (np.ndarray(float)): range of scales used to compute the CD
        cond (bool): whether distances are saved in the condensed form (set to False for point in continuum spaces).
        plot (bool): whether to plot ID vs scale

    Returns:
        scales (np.array(int or float)): size of boxes used to cover the dataset
        ids (np.array(float)): intrinsic dimensions found at different scales
    """

    N = len(dists)
    CI = []
    ids = []

    for i, r in enumerate(scales):
        if cond:
            ci = np.sum(dists[:, r - 1]) - N
        else:
            ci = np.sum(dists < r) - N

        ci = ci / N / (N - 1)
        if ci < 1e-10:
            ci = 1.0 / N / (N - 1)

        CI.append(ci)

        if i < 1:
            continue

        coeffs = np.polyfit(np.log(scales[: i + 1]), np.log(CI[: i + 1]), 1)
        ids.append(coeffs[0])

    if plot:
        plt.figure()
        plt.scatter(np.log(scales), np.log(CI), s=5)
        plt.plot(
            np.log(scales), np.log(scales ** coeffs[0]) + coeffs[1], color="orange"
        )

        plt.figure()
        plt.plot(scales[1:], ids)

    return ids, scales[1:], CI


# --------------------------------------------------------------------------------------


def _binomial_model_validation(
    k, n, p, artificial_samples=100000, k_bootstrap=20, plot=False
):
    """Perform the model validation for the binomial estimator. To this aim, an artificial set of binomially distributed
    points is extracted and compared to the observed ones. The quantitative test should be performed by means of the 2-samples Epps-Singleton tests,
    as it is capable of dealing with discrete distribution. However, the scipy routine happens to have computational issues, with SVD often not converging.
    For this reason we stick to the 2-samples Kolmogorov-Smirnoff test that, even if it is supposed to operate on continuous distributions,
    it still provides reasonable and interpretable results.
    The associated statistics and p-value are returned.

    Args:
        k (int or np.ndarray(int)): Observed points in outer shells.
        n (np.ndarray(int)): Observed points in the inner shells.
        p (float): Tested Binomial parameter
        artificial_samples (int, default=1000000): number of theoretical samples to be compared with the observed ones.
        k_bootstrap (int, default=1): number of bootstrap resampling in order to obtain more reliable p-values
        plot (bool, default=False): flag that, if se to True, allows to plot the observed vs theoretical distributions

    Returns:
        ks_statistics (float): max distance between cumulative distribution functions
        p_value (float): p-value obtained from the test
    """
    assert 0 < p < 1, "The binomial parameter must be in the interval (0,1)"
    # sample an artificial ensemble of binomially distributed points, to be compared with the observed ones
    if n.shape[0] < artificial_samples:
        if isinstance(k, np.ndarray):  # id computed found with inhomogenous k
            replicas = int(artificial_samples / n.shape[0])
            n_samp = np.array(
                [rng.binomial(ki, p, size=replicas) for ki in (k - 1)]
            ).reshape(-1)
        else:  # id sampled with constant k
            n_samp = rng.binomial(k - 1, p, size=artificial_samples)
    else:
        if isinstance(k, np.ndarray):
            n_samp = rng.binomial(k - 1, p)
        else:
            n_samp = rng.binomial(k - 1, p, size=n.shape[0])

    if plot:
        plot_cdf(n - 1, n_samp)

    # es_d, es_pv = es2s(n_samp, n - 1)
    ks_d, ks_pv = ks_2samp(n_samp, n - 1)

    # possibly test against re-sampled distribution using bootstrap
    if k_bootstrap > 1:
        kss = [ks_d]
        pvs = [ks_pv]
        for ki in range(k_bootstrap):
            n_temp = rng.choice(n, size=len(n), replace=True)
            # es_d, es_pv = es2s(n_samp, n_temp - 1)
            ks_d, ks_pv = ks_2samp(n_samp, n_temp - 1)
            kss.append(ks_d)
            pvs.append(ks_pv)
        return np.mean(kss), np.mean(pvs)
    else:
        return ks_d, ks_pv
