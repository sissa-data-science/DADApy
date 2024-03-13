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

from scipy.stats import epps_singleton_2samp as es2s
from scipy.stats import ks_2samp as ks2s
from scipy.stats import cramervonmises_2samp as cvm2s
from scipy.stats import chisquare as x2
from scipy.stats import binom
from scipy.stats import kstwo
import os

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

    return ids, scales[1:], CI

# --------------------------------------------------------------------------------------


def _binomial_model_validation(k, n, p, artificial_samples=100000, k_bootstrap=100, plot=True):
    """Perform the model validation for the binomial estimator. To this aim, an artificial set of binomially distributed
    points is extracted and compared to the observed ones. The quantitative test is performed by means of the .... test.
    The associated statistics and p-value is returned

    Args:
        k (int or np.ndarray(int)): Observed points in outer shells.
        n (np.ndarray(int)): Observed points in the inner shells.
        p (float): Tested Binomial parameter
        artificial_samples (int, default=1000000): number of theoretical samples to be compared with the observed ones.

    Returns:
        statistics (float): test statistics
        p_value (float): p-value obtained from the test
    """
    assert 0 < p < 1, "The binomial parameter must be in the interval (0,1)"
    # sample an artificial ensemble of binomially distributed points to be compared with the observed ones
    if n.shape[0] < artificial_samples:
        if isinstance(k, np.ndarray):
            replicas = int(artificial_samples / n.shape[0])
            n_samp = np.array([rng.binomial(ki, p, size=replicas) for ki in (k-1)]).reshape(-1)
        else:
            n_samp = rng.binomial(k-1, p, size=artificial_samples)
    else:
        if isinstance(k, np.ndarray):
            n_samp = rng.binomial(k-1, p)
        else:
            n_samp = rng.binomial(k-1, p, size=n.shape[0])
                                    
    # compute the theoretical probabilities
#    if isinstance(k, np.ndarray):
#        k_sup = k.max()
#        p_k = np.array([sum(k-1 == i) for i in range(0, k_sup)]) / len(k)
#        p_theo = np.sum([ binom.pmf(range(0, k_sup), ki, p)*p_k[ki] for ki in range(0, k_sup) ], axis=0)
#    else:
#        k_sup = k
#        p_theo = binom.pmf(range(0, k), k-1, p)

    # observed probability distribution
    #p_obs = np.array([sum(n-1 == i) for i in range(0, k_sup)]) / len(n)
#   commented out as it takes long time and is not needed at the moment
#    print('computing p_samp...')
#    p_samp = np.array([sum(n_samp == i) for i in range(0,k_sup)]) / len(n_samp)
    
    #if plot:
    #    print('plotting...')
    #    plt.figure()
    #    plt.plot(p_theo, label='p theo | id')
    #    plt.plot(p_samp, label='p samp | id')
    #    plt.plot(p_obs, label='p obs')
    #    plt.xlabel('n',fontsize=14)
    #    plt.ylabel('p(n)',fontsize=14)
    #plt.yscale('log')
    #plt.show()
    
    #x2_d1, x2_pv1 = x2(p_obs, p_theo, ddof=4)
    #x2_d = np.array([x2(p_obs, p_theo, ddof=ki) for ki in range(1, k_sup)])
    
    #x2_d, x2_pv = x2(len(n)*p_obs, len(n)*p_theo, ddof=1)
    #f_theo = len(n)*p_theo
    #mask = f_theo > 7
    #chi2 = sum((p_obs-p_theo)**2/p_theo)*len(n)
    #chi2 = sum((f_obs[mask]-f_theo[mask])**2/f_theo[mask])
    
    # test against sampled distribution using bootstrap (cramer von mises is left out)
    pvs = np.zeros((k_bootstrap,2))#3
    for ki in range(k_bootstrap):
        n_temp = rng.choice(n,size=len(n),replace=True)
        print(ki,end='\r')
        #p_temp = np.array([sum(n_temp-1 == i) for i in range(0, k_sup)]) / len(n)
        
        #if plot:
        #    if ki==0:
        #        plt.plot(p_temp, color='grey',alpha=0.25,zorder=-1,label='p bootstrap')
        #    else:
        #        plt.plot(p_temp, color='grey',alpha=0.25,zorder=-1)
                
        ks_d, ks_pv = ks2s(n_samp, n_temp-1)
        es_d, es_pv = es2s(n_samp, n_temp-1)#, t=(0.45, 0.55))
        #appo = cvm2s(n_samp, n_temp-1)
        #cvm_d, cvm_pv = appo.statistic,appo.pvalue
        pvs[ki] = (ks_pv,es_pv)#,cvm_pv)

    #if plot:
    #    plt.legend()
    #    plt.show()
    
    ave = np.array([pvs[:i*5].mean(axis=0) for i in range(1,k_bootstrap//5+1)])
    median = np.array([np.median(pvs[:i*5],axis=0) for i in range(1,k_bootstrap//5+1)])
    
    if plot:
        labels=['ks','es','cvm']
        plt.figure()
        [plt.plot(range(5,k_bootstrap+5,5),ave[:,j],label=labels[j]) for j in range(2)]
        [plt.plot(range(5,k_bootstrap+5,5),median[:,j],label=labels[j]) for j in range(2)]
        plt.xlabel('# of bootstrap',fontsize=14)
        plt.ylabel('average pv',fontsize=14)
        plt.yscale('log')
        plt.legend()
        plt.show()
    
    # test against theoretical distribution 
    #ks_stat = max(abs(np.cumsum(p_obs)-np.cumsum(p_theo)))
    #ks_p_value = kstwo.sf(ks_stat, len(n))
    

    #print("KS\t", ks_d, ks_pv)
    #print("KS_hm\t", ks_stat, p_value)
    #print("X2 auto\t", x2_d, x2_pv)
    #print("X2 hm\t", x2_d1, x2_pv1)
    # print("X2:\t", x2_d, x2_pv)
    # print("ES:\t", es_d, es_pv)
    # return ks_d, ks_pv
    return ave[-1,0],ave[-1,1], median[-1,0],median[-1,1]
