from jax import numpy as jnp, jit, lax, random, grad
from jax.tree_util import register_pytree_node
from jax.experimental.host_callback import call
from jax.scipy.special import gammaln
from jax import devices as jdevices
import sys, os
from dadapy.hamming import Hamming
import numpy as np
from time import time

# from jax.config import config
# config.update('jax_platform_name', 'cpu')
# config.update("jax_enable_x64", True)
# config.config_with_absl()
jdevices("cpu")[0]
eps = 1e-7


class Optimizer:
    """ """

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

    ### This is necessary to keep the last *accepted* move
    Op.d0_r = Op.d0
    Op.d1_r = Op.d1
    Op = compute_Pmodel(Op)
    Op = compute_KLd(Op)
    Op.acc_ratio = jnp.double(Op.accepted) / jnp.double(Op.Nsteps)
    return Op


def test_initial_condition(Op, d0, d1):
    """ """
    Op.d0_r = jnp.double(d0)
    Op.d1_r = jnp.double(d1)
    Op = compute_Pmodel(Op)
    Op = compute_KLd(Op)
    return np.log(Op.KL)


class BID:
    def __init__(
        self,
        H=None,
        alphamin=0.0,
        alphamax=0.2,
        seed=1,
        d00=jnp.double(0),
        d10=jnp.double(0),
        delta=5e-4,
        Nsteps=1e6,
        optfolder0=f"results/opt/",
        load_initial_condition_flag=False,
        optimization_elapsed_time=None,
        export_logKLs=0,  # To export the curve of logKLs during optimization
        L=0,  # Number of bits / Ising spins
    ):
        self.H = H
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
        self.optfile = self.optfolder + f"opt.txt"
        self.valfile = self.optfolder + f"model_validation.txt"
        self.KLfile = self.optfolder + f"logKLs_opt.txt"

    def set_idmin(
        self,
    ):
        if self.regularize == False:
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
                test_initial_condition(
                    self.Op,
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

        print(f"starting optimization")
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
