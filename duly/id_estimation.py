from duly._base import *
from scipy.optimize import curve_fit
import duly.utils as ut

class IdEstimation(Base):

    def __init__(self, coordinates=None, distances=None, maxk=None, verbose=False, njobs=cores):
        super().__init__(coordinates=coordinates, distances=distances, maxk=maxk, verbose=verbose,
                         njobs=njobs)

    def compute_id_diego(self, nneigh=1, fraction=0.95, d0=0.1, d1=1000):
        assert (self.distances is not None)
        if self.verb: print('ID estimation started, using nneigh = {}'.format(nneigh))

        # remove the 5% highest value of mu
        mu_full = self.distances[:, 2 * nneigh] / self.distances[:, 1 * nneigh]
        mu_full.sort()
        Nele_eff = int(len(mu_full) * fraction)
        mu = mu_full[:Nele_eff]

        f1 = ut._f(d1, mu, nneigh, Nele_eff)
        while (abs(d0 - d1) > 1.e-5):
            d2 = (d0 + d1) / 2.
            f2 = ut._f(d2, mu, nneigh, Nele_eff)
            if f2 * f1 > 0:
                d1 = d2
            else:
                d0 = d2

        d = (d0 + d1) / 2.

        self.id_selected = int(d)
        if self.verb: print('ID estimation finished, id selected = {}'.format(self.id_selected))

    def compute_id(self, decimation=1, fraction=0.9, algorithm='base'):
        assert (0. < decimation and decimation <= 1.)
        # self.id_estimated_ml, self.id_estimated_ml_std = return_id(self.distances, decimation)
        # Nele = len(distances)
        # remove highest mu values
        Nele_eff = int(self.Nele * fraction)
        mus = np.log(self.distances[:, 2] / self.distances[:, 1])
        mus = np.sort(mus)[:Nele_eff]
        Nele_eff_dec = int(np.around(decimation * Nele_eff, decimals=0))
        idxs = np.arange(Nele_eff)
        idxs = np.random.choice(idxs, size=Nele_eff_dec, replace=False, p=None)
        mus_reduced = mus[idxs]
        if algorithm == 'ml':
            id = Nele_eff_dec / np.sum(mus_reduced)
        elif algorithm == 'base':
            def func(x, m):
                return m * x

            y = np.array([-np.log(1 - i / self.Nele) for i in range(1, Nele_eff + 1)])
            y = y[idxs]
            id, _ = curve_fit(func, mus_reduced, y)
        self.id_selected = np.around(id, decimals=2)[0]
        if self.verb:
            # print('ID estimated from ML is {:f} +- {:f}'.format(self.id_estimated_ml, self.id_estimated_ml_std))
            # print(f'Selecting ID of {self.id_selected}')
            print(f'ID estimation finished: selecting ID of {self.id_selected}')

    def compute_id_gammaprior(self, alpha=2, beta=5):
        if self.distances is None: self.compute_distances()

        if self.verb: print(
            'ID estimation started, using alpha = {} and beta = {}'.format(alpha, alpha))

        distances_used = self.distances

        sum_log_mus = np.sum(np.log(distances_used[:, 2] / distances_used[:, 1]))

        alpha_post = alpha + self.Nele
        beta_post = beta + sum_log_mus

        mean_post = alpha_post / beta_post
        std_post = np.sqrt(alpha_post / beta_post ** 2)
        mode_post = (alpha_post - 1) / beta_post

        self.id_alpha_post = alpha_post
        self.id_beta_post = beta_post
        self.id_estimated_mp = mean_post
        self.id_estimated_mp_std = std_post
        self.id_estimated_map = mode_post
        self.id_selected = int(np.around(self.id_estimated_mp, decimals=0))

    def set_id(self, id):
        self.id_selected = id