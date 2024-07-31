import os
from time import time

import numpy as np
from jax import devices as jdevices
from jax import jit, lax
from jax import numpy as jnp
from jax import random
from jax.scipy.special import gammaln
from jax.tree_util import register_pytree_node

# from jax.experimental.host_callback import call

jdevices("cpu")[0]  # to run JAX on CPU
eps = 1e-7  # good old small epsilon


class Hamming:
    def __init__(
        self,
        q=2,  # number of states: 2 for binary spins. In the future we can think of extending this to q>2.
        coordinates=None,  # spins: must be normalized to +-1 to compute distances.
        distances=None,  #
        crossed_distances=0,  # 0 means we have one dataset with N samples and N(N-1)/2 (correlated) distances.
        verbose=False,  #
    ):
        self.q = q
        self.coordinates = coordinates
        self.metric = "hamming"
        self.distances = distances
        self.crossed_distances = crossed_distances
        self.verbose = verbose

        self.r = None
        self.r_idx = None
        self.D_values = None
        self.D_counts = None
        self.D_probs = None
        self.D_mu_emp = None
        self.D_var_emp = None

    def compute_distances(
        self,
        sort=False,
        check_format=True,
    ):
        """
        Computes all to all distances in dataset and stores them in the matrix self.distances
        """
        if self.q == 2:
            self.distances = jcompute_distances(
                X1=self.coordinates,
                X2=self.coordinates,
                crossed_distances=self.crossed_distances,
                check_format=check_format,
                sort=sort,
            )

    """TODO: MODIFY HISTOGRAM ROUTINE TO DISCARD THE TRIVIAL ZEROS WHEN CROSSED_DISTANCES = 1"""
    def D_histogram(
        self,
        compute_flag=0,  # 1 to compute histogram (else it is loaded)
        save=False,  # 1 to save computed histogram
        resultsfolder="results/hist/",
    ):
        """
        Given the computed distances this routine computes the histogram (Pemp) and saves is
        """
        assert self.crossed_distances == 0
        
        if save:
            os.makedirs(resultsfolder, exist_ok=True)
        c_fname = resultsfolder + "counts"

        if compute_flag:
            self.D_values, self.D_counts = np.unique(self.distances, return_counts=True)

            if self.crossed_distances == 0:
                Nsamples = self.distances.shape[0]
                assert self.D_values[0] == 0  # trivial zeros
                Nzeros = int(
                    Nsamples * (Nsamples + 1) / 2
                )  # trivial zeros, Gauss sum of them
                self.D_counts[0] -= Nzeros
                if self.D_counts[0] == 0:
                    self.D_values = self.D_values[1:]
                    self.D_counts = self.D_counts[1:]
            self.D_probs = self.D_counts / np.sum(self.D_counts)

            if save:
                np.savetxt(
                    fname=c_fname + ".txt",
                    X=np.transpose([self.D_values, self.D_counts]),
                    fmt="%d,%d",
                )
        else:
            f = np.loadtxt(c_fname + ".txt", delimiter=",", dtype=int)
            self.D_values = f[:, 0]
            self.D_counts = f[:, 1]
            self.D_probs = self.D_counts / np.sum(self.D_counts)

    def set_r_quantile(self, alpha, round=True, precision=10):
        if round:
            alpha = np.round(alpha, precision)
            self.D_probs = np.round(self.D_probs, precision)

        indices = np.where(np.cumsum(self.D_probs) <= alpha)[0]
        if len(indices) == 0:
            self.r_idx = 0
        else:
            self.r_idx = indices[-1]

        self.r = int(self.D_values[self.r_idx])
        return

    def set_r(self, r=None, n_sigma=3):
        if r is not None:
            self.r = r
        else:
            self.compute_moments()
            r = int(np.round(self.D_mu_emp - n_sigma * np.sqrt(self.D_var_emp), 0))
            if r >= self.D_values[0]:
                self.r = r
            else:
                self.r = self.D_values[0]

    def set_r_idx(self, r_idx=None):
        if r_idx is not None:
            self.r_idx = r_idx
        else:
            aux = np.asarray(self.r == self.D_values).nonzero()[0]
            if len(aux) > 0:
                self.r_idx = aux[0]
            else:
                self.r_idx = 0

    def compute_moments(self):
        self.D_mu_emp = np.dot(self.D_probs, self.D_values)
        self.D_var_emp = np.dot(self.D_probs, self.D_values**2) - self.D_mu_emp**2


def check_data_format(X):
    e1, e2 = np.unique(X)
    assert (
        e1 == -1 and e2 == 1
    ), f"spins have to be formatted to -+1, but {np.unique(X)=}"


def jcompute_distances(
    X1,
    X2,
    crossed_distances,
    check_format=True,
    sort=False,
):
    """This routine works for Ising spins variables defined as +-1 (this is faster than scipy)"""
    X1 = jnp.array(X1).astype(jnp.int32)
    X2 = jnp.array(X2).astype(jnp.int32)

    if check_format:
        check_data_format(X1)
        check_data_format(X2)

    Ns1, N = X1.shape
    Ns2 = X2.shape[0]  # the samples in the other dataset must have also N spins...

    distances = jnp.zeros(shape=(Ns1, Ns2), dtype=jnp.int32)
    sample_idx = 0
    lower_idx = 0
    pytree = {
        "crossed_distances": crossed_distances,
        "D": distances,
        "X1": X1,
        "X2": X2,
        "sample_idx": sample_idx,
        "lower_idx": lower_idx,  # to avoid computing distances twice if crossed_distances=1
        "Ns1": Ns1,  # number of samples in dataset 1
        "Ns2": Ns2,  # number of samples in dataset 2
        "N": N,  # number of spins in each sample
    }

    pytree = lax.fori_loop(
        lower=0, upper=Ns1, body_fun=_jcompute_distances, init_val=pytree
    )

    if sort:
        return np.array(jnp.sort(pytree["D"]))
    else:
        return np.array(pytree["D"])


@jit
def _jcompute_distances(idx, pytree):
    """
    for each data sample indexed by "sample_idx" (row), computes the distance between it and the rest
    """
    pytree["sample_idx"] = idx
    pytree = lax.cond(
        pytree["crossed_distances"], _set_lower_idx_true, _set_lower_idx_false, pytree
    )
    pytree = lax.fori_loop(
        lower=pytree["lower_idx"],
        upper=pytree["Ns2"],
        body_fun=compute_row_distances,
        init_val=pytree,
    )
    return pytree


def _set_lower_idx_true(pytree):
    """
    if we have two datasets, we have Ns1 * Ns2 distances to compute. 
    """
    pytree["lower_idx"] = 0
    return pytree


def _set_lower_idx_false(pytree):
    """
    if we have one dataset, we have Ns(Ns-1)/2 distances to compute (the upper triangular part of "distances")
    """
    pytree["lower_idx"] = pytree["sample_idx"] + 1
    return pytree


@jit
def compute_row_distances(_idx, pytree):
    """
    for each data sample indexed by "sample_idx", computes the distance between it and the rest
    """
    pytree["D"] = (
        pytree["D"]
        .at[pytree["sample_idx"], _idx]
        .set(
            jnp.int32(
                (
                    pytree["N"]
                    - jnp.dot(
                        pytree["X1"][pytree["sample_idx"], :], pytree["X2"][_idx, :]
                    )
                )
                / 2
            )
        )
    )
    return pytree


class Optimizer:
    """
    Stochastic optimization
    """

    def __init__(
        self,
        key=0,
        d0=0.0,  # BID
        d0_r=0.0,  # BID + random perturbation (*** used by compute_Pmodel instead of d0...)
        d1=0.0,  # slope
        d1_r=0.0,  # slope + random pertubation (*** used by compute_Pmodel instead of d1...)
        delta=0.0,  # optimization step size
        KL=jnp.double(0.0),  # KL divergence between Pemp(r) and Pmodel(r)
        KL_aux=jnp.inf,  # auxiliary variable to check when KL decreases
        remp=None,  # vector with empirical Hamming distances
        Pemp=None,  # vector with empirical probabilities
        Pmodel=None,  # vector with model probabilities
        Nsteps=0,  # Number of steps for the optimization
        accepted=0,  # Accepted moves
        acc_ratio=jnp.double(1.0),  # Acceptance ratio
        save_logKLs_flag=0,  # Flag to save logKLs during optimization
        logKLs=None,  # Vector with logKLs during optimization
        idx=0,  # Auxiliary index
        mod_divisor=0,  # Auxiliary variable to export the log KL
        Nsteps_max=1000,  # Total number of saved steps (subsample of the total number of steps)
    ):
        self.key = key
        self.d0 = d0
        self.d0_r = d0_r
        self.d1 = d1
        self.d1_r = d1_r
        self.delta = delta
        self.KL = KL
        self.KL_aux = KL_aux
        self.remp = remp
        self.Pemp = Pemp
        self.Pmodel = Pmodel
        self.Nsteps = Nsteps
        self.accepted = accepted
        self.acc_ratio = acc_ratio
        self.save_logKLs_flag = save_logKLs_flag
        self.logKLs = logKLs
        self.idx = idx
        self.mod_divisor = mod_divisor
        self.Nsteps_max = Nsteps_max

    def _tree_flatten(self):
        children = (
            self.key,
            self.d0,
            self.d0_r,
            self.d1,
            self.d1_r,
            self.delta,
            self.KL,
            self.KL_aux,
            self.remp,
            self.Pemp,
            self.Pmodel,
            self.Nsteps,
            self.accepted,
            self.acc_ratio,
            self.save_logKLs_flag,
            self.logKLs,
            self.idx,
            self.mod_divisor,
        )  # arrays / dynamic values
        aux_data = {
            "Nsteps_max": self.Nsteps_max,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


register_pytree_node(Optimizer, Optimizer._tree_flatten, Optimizer._tree_unflatten)


def _compute_Pmodel(idx, Op):
    """
    note that this routine uses Op.d0_r and Op.d1_r to compute Pmodel
    """
    ID = Op.d0_r + Op.d1_r * Op.remp[idx]
    Op.Pmodel = Op.Pmodel.at[idx].set(
        jnp.exp(
            gammaln(ID + jnp.double(1))
            - gammaln(Op.remp[idx] + 1)
            - gammaln(ID - Op.remp[idx] + 1)
            - ID * jnp.log(jnp.double(2))
        )
    )
    #  call(lambda x: print(f'{x}'),Op.Pmodel[idx])
    return Op


@jit
def compute_Pmodel(Op):
    Op = lax.fori_loop(
        lower=0, upper=Op.Pmodel.shape[0], body_fun=_compute_Pmodel, init_val=Op
    )
    Op.Pmodel /= jnp.sum(Op.Pmodel)
    return Op


@jit
def step(idx, Op):
    Op.key, subkey = random.split(Op.key, num=2)
    r = random.uniform(subkey, dtype=jnp.float64)
    Op.d0_r = Op.d0 * (1 + Op.delta * (r - jnp.double(0.5)))

    Op.key, subkey = random.split(Op.key, num=2)
    rr = random.uniform(subkey, dtype=jnp.float64)
    Op.d1_r = Op.d1 * (1 + Op.delta * (rr - jnp.double(0.5)))

    Op = compute_Pmodel(Op)
    Op = compute_KLd(Op)
    Op = lax.cond(Op.KL <= Op.KL_aux, update_state, do_nothing, Op)
    logical_condition = jnp.logical_and(
        Op.save_logKLs_flag, jnp.mod(idx, Op.mod_divisor) == 0
    )
    Op = lax.cond(logical_condition, save_logKL, do_nothing, Op)
    return Op


@jit
def compute_KLd(Op):
    Op.KL = jnp.sum(Op.Pemp * jnp.log(Op.Pemp / Op.Pmodel))
    return Op


@jit
def update_state(Op):
    Op.d0 = Op.d0_r
    Op.d1 = Op.d1_r
    Op.KL_aux = Op.KL
    Op.accepted += 1
    return Op


@jit
def do_nothing(Op):
    return Op


@jit
def save_logKL(Op):
    Op.logKLs = Op.logKLs.at[Op.idx].set(jnp.log(Op.KL))
    Op.idx += 1
    return Op


@jit
def minimize_KL(Op):
    Op.logKLs = jnp.empty(shape=(Op.Nsteps_max), dtype=jnp.double)
    Op.mod_divisor = Op.Nsteps // Op.Nsteps_max

    Op = lax.fori_loop(lower=0, upper=Op.Nsteps, body_fun=step, init_val=Op)

    # This is necessary to keep the last *accepted* move
    Op.d0_r = Op.d0
    Op.d1_r = Op.d1
    Op = compute_Pmodel(Op)
    Op = compute_KLd(Op)
    Op.acc_ratio = jnp.double(Op.accepted) / jnp.double(Op.Nsteps)
    return Op


class BID:
    def __init__(
        self,
        H=None,  # instance of Hamming class defined above
        Op=None,  # instance of Optimizer class defined above
        alphamin=0.0,
        alphamax=0.2,
        seed=1,
        d00=jnp.double(0),
        d10=jnp.double(0),
        delta=5e-4,
        Nsteps=1e6,
        optfolder0="results/opt/",
        load_initial_condition_flag=False,
        optimization_elapsed_time=None,
        export_logKLs=0,  # To export the curve of logKLs during optimization
        L=0,  # Number of bits / Ising spins
    ):
        self.H = H
        self.Op = Op
        self.alphamin = alphamin
        self.alphamax = alphamax
        self.seed = seed
        self.d00 = d00
        self.d10 = d10
        self.delta = delta
        self.Nsteps = Nsteps
        self.optimization_elapsed_time = optimization_elapsed_time  # in minutes
        self.export_logKLs = export_logKLs
        self.L = L

        self.key0 = random.PRNGKey(self.seed)
        self.optfolder0 = optfolder0
        self.load_initial_condition_flag = load_initial_condition_flag

        if np.isclose(alphamin, 0):
            self.regularize = False
        else:
            self.regularize = True

    def load_initial_condition(
        self,
    ):
        self.set_filepaths()
        _, self.d00, self.d10, _ = self.load_results()

    def set_filepaths(
        self,
    ):
        self.optfolder = self.optfolder0
        self.optfolder += f"alphamin{self.alphamin:.5f}/"
        self.optfolder += f"alphamax{self.alphamax:.5f}/"
        self.optfolder += f"Nsteps{self.Nsteps}/"
        self.optfolder += f"delta{self.delta:.5f}/"
        self.optfolder += f"seed{self.seed}/"
        self.optfile = self.optfolder + "opt.txt"
        self.valfile = self.optfolder + "model_validation.txt"
        self.KLfile = self.optfolder + "logKLs_opt.txt"

    def set_idmin(
        self,
    ):
        if self.regularize is False:
            self.idmin = 0
            self.rmin = self.H.D_values[0]
        else:
            self.H.set_r_quantile(self.alphamin)
            self.rmin = self.H.r
            self.idmin = self.H.r_idx
            self.H.r = None
            self.H.r_idx = None

    def set_idmax(
        self,
    ):
        self.H.set_r_quantile(self.alphamax)
        self.rmax = self.H.r
        self.idmax = self.H.r_idx
        self.H.r = None
        self.H.r_idx = None

    def truncate_hist(self):
        self.remp = jnp.array(
            self.H.D_values[self.idmin : self.idmax + 1], dtype=jnp.float64
        )
        self.Pemp = jnp.array(
            self.H.D_probs[self.idmin : self.idmax + 1], dtype=jnp.float64
        )
        self.Pemp /= jnp.sum(self.Pemp)
        self.Pmodel = jnp.zeros(shape=self.Pemp.shape, dtype=jnp.float64)

    def test_initial_condition(self, d0, d1):
        self.Op.d0_r = jnp.double(d0)
        self.Op.d1_r = jnp.double(d1)
        self.Op = compute_Pmodel(self.Op)
        self.Op = compute_KLd(self.Op)
        return np.log(self.Op.KL)

    def set_initial_condition(self, d00min=0.05, d00max=0.95, d00step=0.05):
        # our home-made guess:
        d00_guess_list = jnp.array([jnp.double(self.Op.remp[-1])])
        d10_guess_list = jnp.array([jnp.double(1)])

        if self.L != 0:
            # Inspired by Cristopher Erazo:
            self.H.compute_moments()
            _d00_guess_list = self.L * jnp.arange(
                d00min, d00max + eps, d00step, dtype=jnp.double
            )
            _d10_guess_list = jnp.double(2) - _d00_guess_list / jnp.double(
                self.H.D_mu_emp
            )

            d00_guess_list = jnp.concatenate((d00_guess_list, _d00_guess_list))
            d10_guess_list = np.concatenate((d10_guess_list, _d10_guess_list))

        logKLs0 = jnp.empty(shape=(len(d00_guess_list)), dtype=jnp.double)
        for i in range(len(d00_guess_list)):
            logKLs0 = logKLs0.at[i].set(
                self.test_initial_condition(
                    d00_guess_list[i],
                    d10_guess_list[i],
                )
            )
        # print(f'{logKLs0=}')
        i0 = jnp.nanargmin(logKLs0)
        # print(f'{i0=}')
        # print(f'{logKLs0[i0]=}')
        self.d00 = d00_guess_list[i0]  # ; print(f'{self.d00=}')
        self.d10 = d10_guess_list[i0]  # ; print(f'{self.d10=}')
        self.Op.d0 = self.d00
        self.Op.d1 = self.d10
        self.Op.KL_aux = jnp.exp(logKLs0[i0])

    def computeBID(
        self,
    ):
        self.set_idmin()
        self.set_idmax()
        self.truncate_hist()
        self.set_filepaths()
        os.makedirs(self.optfolder, exist_ok=True)

        self.Op = Optimizer(
            key=self.key0,
            d0=jnp.double(self.d00),
            d1=jnp.double(self.d10),
            delta=jnp.double(self.delta),
            remp=self.remp,
            Pemp=self.Pemp,
            Pmodel=self.Pmodel,
            Nsteps=self.Nsteps,
            save_logKLs_flag=self.export_logKLs,
        )
        self.set_initial_condition()

        print("starting optimization")
        starting_time = time()
        self.Op = minimize_KL(self.Op)
        self.optimization_elapsed_time = (time() - starting_time) / 60.0
        print(f"optimization took {self.optimization_elapsed_time:.1f} minutes")
        print(
            f"d_0={self.Op.d0:.3f},d_1={self.Op.d1:.3f},logKL={jnp.log(self.Op.KL):.2f}"
        )

        os.system(f"rm -f {self.optfile}")
        print(
            f"{self.rmax:d},{self.Op.d0:.8f},{self.Op.d1:8f},{np.log(self.Op.KL):.8f}",
            file=open(self.optfile, "a"),
        )
        np.savetxt(
            fname=self.valfile, X=np.transpose([self.remp, self.Pemp, self.Op.Pmodel])
        )
        if self.export_logKLs:
            np.savetxt(fname=self.KLfile, X=self.Op.logKLs)

    def load_results(
        self,
    ):
        self.set_filepaths()
        return np.loadtxt(self.optfile, unpack=True, delimiter=",")

    def load_fit(
        self,
    ):
        self.set_filepaths()
        return np.loadtxt(self.valfile, unpack=True)

    def load_logKLs_opt(
        self,
    ):
        self.set_filepaths()
        return np.loadtxt(self.KLfile)
