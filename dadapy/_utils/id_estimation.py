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
            # bin_edges = np.array([np.hstack([inf - 1e-5 - offset, x - 1e-5 - offset])
            #        for x in bin_temp[:, 1:]])
            bin_edges = np.array(
                [
                    np.hstack([inf - scale + offset + 1e-5, x + offset + 1e-5])
                    for x in bin_temp
                ]
            )
            # ind = np.where((bin_edges[0] < sup - 1) == False)[0][0]
            # bin_edges = bin_edges[:, : ind + 1]
            # print(bin_edges)

            H1, e = np.histogramdd(data, bins=bin_edges)
            # print(e[0],e[0][0],e[0][-1])
            # if verb:
            #    print(H1.sum())
            #    if int(H1.sum()) != array.shape[0]:
            #       print('for scale ', scale, ' and offset ', offset, ' the covering is badly shaped')
            # touched.append(np.inf)
            # continue
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
            label=f"Fit: {np.round(coeffs[0],3)}X+{coeffs[1]}",
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

    return ids, scales[1:]
