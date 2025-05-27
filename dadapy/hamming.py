import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import  lax, vmap
from jax import jacfwd, grad, value_and_grad
from jax import random, debug
from jax.scipy.special import gammaln
from flax import struct
from typing import Any
from time import time


eps = 1e-7  # good old small epsilon

### TO RUN ON CPU, before any JAX importing: 
#os.environ["JAX_PLATFORMS"] = "cpu"

### HAMMING CLASS

class Hamming:
    def __init__(
        self,
        q=2,                     # number of states: 2 for binary spins. In the future we can think of extending this to q>2.
        coordinates=None,        # spins: must be normalized to +-1 to compute distances.
        distances=None,  
        crossed_distances=None,  
        verbose=True,  #
    ):
        self.q = q
        self.coordinates = coordinates
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

        self._set_cross_distances()

    def _set_cross_distances(self,):
        """
        CPU: self.cross_distances = 0 corresponds we have one dataset with N_s samples and N_s(N_s-1)/2 distances.
        GPU: self.cross_distances = 1 corresponds to split the dataset in two parts, each with N1 and N2 samples,
        having in total N1*N2 distances.
        TODO: solve allocation memory issues in CPU when self.cross_distances = 1.

        """
        if 'JAX_PLATFORMS' not in os.environ:
            self.crossed_distances = 1
            if self.verbose:
                print(f'running JAX on GPU') 
        else:
            os.environ["JAX_PLATFORMS"] = "cpu"
            self.crossed_distances = 0
            if self.verbose:
                print(f'running JAX on CPU')
        return

    def compute_distances(
        self,
        sort=False,
        check_format=True,
    ):
        """
        Computation of distances runs extremely faster on GPU. (not the case for the BID optimization, though).
        """
        if self.q == 2 and self.crossed_distances == 0:
            self.distances = jcompute_distances(
                X1=self.coordinates,
                X2=self.coordinates,
                crossed_distances=self.crossed_distances,
                check_format=check_format,
                sort=sort,
            )
        elif self.crossed_distances == 1:
            self.X1,self.X2 = subsample_data(self.coordinates)
            self.distances = jcompute_crossed_distances(
                boolean_hamming_distance,
                self.X1,
                self.X2,
            )

    def D_histogram(
        self,
        compute_flag=0,  # 1 to compute histogram (else it is loaded)
        save=False,  # 1 to save computed histogram
        resultsfolder="results/hist/",
        filename="counts.txt",
    ):
        """
        Given the computed distances, this routine computes the histogram (Pemp).
        It defines
        - self.D_values, a vector containing the sampled distances
        - self.D_counts, a vector containing how many times each distance was sampled
        - self.D_probs, self.counts normalized by the total number of counts observed.
        """

        if save:
            os.makedirs(resultsfolder, exist_ok=True)
        _filename = resultsfolder + filename

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
                    fname=_filename,
                    X=np.transpose([self.D_values, self.D_counts]),
                    fmt="%d,%d",
                )
        else:
            f = np.loadtxt(_filename, delimiter=",", dtype=int)
            self.D_values = f[:, 0]
            self.D_counts = f[:, 1]
            self.D_probs = self.D_counts / np.sum(self.D_counts)

    def set_r_quantile(self, alpha, round=True, precision=10):
        """
        Defines

        - self.r as the quantile of order alpha of self.D_probs,
        which can be used for rmax or rmin,
        to discard distances larger than rmax or smaller than rmin, respectively.
        - self.r_idx as the index of self.r in self.D_values
        """
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

    def compute_moments(self):
        """
        computes the empirical mean and variance of H.D_probs
        """
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
    X1 = jnp.array(X1).astype(jnp.int8)
    X2 = jnp.array(X2).astype(jnp.int8)

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


@jax.jit
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


@jax.jit
def compute_row_distances(_idx, pytree):
    """
    for each data sample indexed by "sample_idx", computes the distance between it and the rest
    """
    pytree["D"] = (
        pytree["D"]
        .at[pytree["sample_idx"], _idx]
        .set(
                (
                    pytree["N"]
                    - jnp.dot(
                        pytree["X1"][pytree["sample_idx"], :].astype(jnp.int32), pytree["X2"][_idx, :].astype(jnp.int32)
                    )
                )
                // 2
        )
    )
    return pytree


def subsample_data(coordinates,N1=None,seed=0):
    if N1 == None:
        N1 = coordinates.shape[0] // 2
    key = random.PRNGKey(seed)
    key, subkey = random.split(key, num=2)

    indices_rows = random.choice(
    subkey, jnp.arange(coordinates.shape[0]), shape=(N1,), replace=False
    )
    indices_columns = jnp.setdiff1d(jnp.arange(coordinates.shape[0]), 
                                    indices_rows, 
                                    assume_unique=True)

    return coordinates[indices_rows],coordinates[indices_columns]

@jax.jit
def boolean_hamming_distance(x, y):
    return jnp.count_nonzero(x != y)

def jcompute_crossed_distances(dist, xs, ys):
    return vmap(lambda x: vmap(lambda y: dist(x, y))(xs))(ys).T

### BID CLASS
class BID:
    def __init__(
        self,
        H=Hamming(),  # instance of Hamming class
        opt=None,  # instance of Optimizer class
        alphamin=.05,
        alphamax=0.4,
        ds=jnp.zeros(shape=2,dtype=jnp.double),
        steps_initial=jnp.array([1E-1],dtype=jnp.double), # amplitude modulating the gradient descent step
        step_final=jnp.double(1e-4),
        Nsteps=jnp.int32(1e3), # total number of optimization steps. There is an early stopping condition, so the optimization can stop before Nsteps.
        n_optimizations = 1, # number of best initial conditions to start optimizations, selecting the best final result
        optfolder0="results/opt/",
        optimization_elapsed_time=None,
        export_results=1,
        export_logKLs=0,  # To export the curve of logKLs during optimization
        L=0,  # Number of bits / Ising spins
    ):
        self.H = H
        self.opt = opt
        self.alphamin = alphamin
        self.alphamax = alphamax
        self.ds = ds
        self.steps_initial = steps_initial
        self.step_final = step_final
        self.Nsteps = Nsteps
        self.n_optimizations = n_optimizations
        self.optimization_elapsed_time = optimization_elapsed_time  # in minutes
        self.export_results = export_results
        self.export_logKLs = export_logKLs
        self.L = L
        self.intrinsic_dim = self.ds[0]

        # self.key0 = random.PRNGKey(self.seed)
        self.optfolder0 = optfolder0

        if np.isclose(alphamin, 0):
            self.regularize = False
        else:
            self.regularize = True

    def set_filepaths(
        self,
    ):
        self.optfolder = self.optfolder0
        self.optfolder += f"alphamin{self.alphamin:.5f}/"
        self.optfolder += f"alphamax{self.alphamax:.5f}/"
        # self.optfolder += f"Nsteps{self.Nsteps}/"
        # self.optfolder += f"step{self.step_initial:.5f}/"
        # self.optfolder += f"seed{self.seed}/"
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

    def initialize_optimizer(self,):
        self.set_idmin()
        self.set_idmax()
        self.truncate_hist()
        self.set_filepaths()
        if self.export_results:
            os.makedirs(self.optfolder, exist_ok=True)

        params = {
                'L':self.L,
                'remp':self.remp,
                'Pemp':self.Pemp,
                'Nsteps':self.Nsteps,
                'steps_initial' : self.steps_initial,
                'step_initial': self.step_initial,
                'step_final' : self.step_final,
                'n_optimizations': self.n_optimizations,
                }
        vars = {
            'indices_optimization': jnp.zeros(shape=(self.n_optimizations,), dtype=jnp.int32),
            }

        opt = Optimizer(params,vars)
        opt = opt.create(params,vars)
        opt = test_initial_conditions(opt)
        return opt


    def computeBID(
        self,
    ):        
        if self.H.verbose == 1:
            print("starting optimization")
        starting_time = time()

        # running the optimization for some different values of the optimization step. 
        opts = []
        logKLs = []
        for step_index,step_initial in enumerate(self.steps_initial):
            self.step_initial = step_initial
            opt = self.initialize_optimizer()
            for idx in opt.vars['indices_optimization']:
                opt = set_initial_condition_idx(idx,opt)
                opt = minimize_loss(opt)
                opts.append(opt)
                logKLs.append(opt.vars['logKL'])

        best_optimization_id = np.nanargmin(jnp.array(logKLs))
        self.opt = opts[best_optimization_id]
        self.opt.vars['ds_optimization'] = self.opt.vars['ds_optimization'][:self.opt.vars['final_optimization_step']]
        self.opt.vars['logKLs'] = self.opt.vars['logKLs'][:self.opt.vars['final_optimization_step']]


        self.optimization_elapsed_time = (time() - starting_time) 

        if self.H.verbose == 1:
            print(f"optimization took {self.optimization_elapsed_time:.1f} sec")
            print(
                f"d_0={self.opt.params['L'] * self.opt.vars['ds'][0]:.3f},d_1={self.opt.vars['ds'][1]:.3f},logKL={self.opt.vars['logKL']:.3f}"
            )

        if self.export_results:
            """ Note that the BID is normalized per bit, so the intrinsic dimension is d0 * L """
            os.system(f"rm -f {self.optfile}")
            print(
                f"{self.rmax:d},{self.opt.params['L'] * self.opt.vars['ds'][0]:.8f},{self.opt.vars['ds'][1]:8f},{self.opt.vars['logKL']:.8f}",
                file=open(self.optfile, "a"),
            )
            np.savetxt(
                fname=self.valfile,
                X=np.transpose([self.remp, 
                                self.Pemp, 
                                self.opt.vars['Pmodel']]),
            )
            if self.export_logKLs:
                np.savetxt(fname=self.KLfile, 
                           X=self.opt.vars['logKLs'])

        self.ds = self.opt.vars['ds']
        self.logKL = self.opt.vars['logKL']
        self.Pmodel = np.array(self.opt.vars['Pmodel'])
        self.intrinsic_dim = self.ds[0] * self.opt.params['L']
        self.d0,self.d1 = self.intrinsic_dim, self.ds[1]

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


### OPTIMIZER

@struct.dataclass
class Optimizer(struct.PyTreeNode):
  params: Any
  vars: Any

  @classmethod
  def create(cls, 
            params,
            vars,
             ):
    params['mean_r'] = jnp.mean(params['remp'])
    params['eps'] = 1e-8  # good-old small epsilon
    vars['ds'] = jnp.zeros(shape=(2),dtype=jnp.double)
    vars['step'] = jnp.double(0)
    vars['grad'] = jnp.zeros(shape=(2),dtype=jnp.double)
    vars['Hessian'] = jnp.zeros(shape=(2,2),dtype=jnp.double)
    vars['logKL'] = jnp.double(0.)
    vars['logPmodel'] = jnp.zeros(shape=params['Pemp'].shape, dtype=jnp.float64)
    vars['Pmodel'] = jnp.zeros(shape=params['Pemp'].shape, dtype=jnp.float64)
    vars['d_of_r'] = jnp.zeros(shape=params['Pemp'].shape, dtype=jnp.float64)

    ### MONITORING OPTIMIZATION
    vars['logKLs'] = jnp.zeros(shape=(params['Nsteps']),dtype=jnp.double)
    vars['ds_optimization'] = jnp.zeros(shape=(params['Nsteps'],len(vars['ds']))) 
    
    ### EARLY STOPPING
    vars['counter_early_stopping'] = jnp.int32(0)
    params['threshold_early_stopping'] = jnp.double(1E-5)
    params['tolerance_early_stopping'] = jnp.int32(200)
    vars['early_stopping_condition'] = False
    vars['final_optimization_step'] = jnp.int32(-1)

    ### INITIAL CONDITION
    params['auxmin'] = jnp.double(0.05)
    params['auxmax'] = jnp.double(0.95)
    params['auxstep'] = jnp.double(0.05)
    params['d00_list'] =  jnp.arange(
                                    params['auxmin'], 
                                    params['auxmax'] + params['eps'], 
                                    params['auxstep'], 
                                    dtype=jnp.double
                                    )
    params['d10_list'] = jnp.double(2) - params['L']  * params['d00_list'] / jnp.dot(params['remp'],params['Pemp']) # smart initial condtion from Cristopher Erazo

    # possibly useless, but previously useful guess:
    params['d10_list'] = jnp.concat([params['d10_list'],
                                    jnp.ones_like(params['d00_list'])]
                                    )
    params['d00_list'] = jnp.concatenate([params['d00_list'],
                                          params['d00_list']])

    vars['logKLs0'] = jnp.empty(shape=params['d00_list'].shape[0], dtype=jnp.double)
    return cls(
              params,
              vars,
              )
  
@jax.jit
def set_initial_condition_idx(idx,opt):

  opt.vars['ds'] = opt.vars['ds'].at[0].set(opt.params['d00_list'][idx % opt.params['d00_list'].shape[0]])
  opt.vars['ds'] = opt.vars['ds'].at[1].set(opt.params['d10_list'][idx % opt.params['d10_list'].shape[0]])
  
  return opt

@jax.jit
def test_initial_conditions(opt):

  opt = lax.fori_loop(lower=0,
                    upper=opt.vars['logKLs0'].shape[0],
                    body_fun=_test_initial_conditions,
                    init_val=opt)

  opt.vars['logKLs0'] = jnp.where(jnp.isnan(opt.vars['logKLs0']), 
                        jnp.inf,
                        opt.vars['logKLs0'], 
                        )
  
  top_k_initial_conditions = lax.top_k(-opt.vars['logKLs0'], 
                                       k=opt.vars['indices_optimization'].shape[0])
  opt.vars['indices_optimization'] = top_k_initial_conditions[1]
  return opt


def _test_initial_conditions(idx,opt):
  opt.vars['ds'] = opt.vars['ds'].at[0].set(opt.params['d00_list'][idx // opt.params['d00_list'].shape[0]])
  opt.vars['ds'] = opt.vars['ds'].at[1].set(opt.params['d10_list'][idx % opt.params['d10_list'].shape[0]])
  opt = compute_KL(opt)
  opt.vars['logKLs0'] = opt.vars['logKLs0'].at[idx].set(opt.vars['logKL'])
  return opt

@jax.jit
def d_of_r(opt,ds):
  opt.vars['d_of_r'] = opt.params['L'] * ds[0] + ds[1] *  opt.params['remp']
  return opt
  
@jax.jit
def _logPmodel(opt):

  opt.vars['logPmodel'] = - opt.vars['d_of_r'] * jnp.log(jnp.double(2))
  opt.vars['logPmodel'] += gammaln(opt.vars['d_of_r']+jnp.double(1))
  opt.vars['logPmodel'] -= gammaln(opt.vars['d_of_r']-opt.params['remp']+jnp.double(1))
  opt.vars['logPmodel'] -= gammaln(opt.params['remp']+jnp.double(1))

  opt.vars['Pmodel'] = jnp.exp(opt.vars['logPmodel'])
  opt.vars['Pmodel'] /= jnp.sum(opt.vars['Pmodel'])
  opt.vars['logPmodel'] = jnp.log(opt.vars['Pmodel'])
  
  return opt


def compute_loss(opt):
  """The KL has two terms, this is the term that depends on the model parameters."""
  opt.vars['loss'] = - jnp.dot(opt.params['Pemp'],opt.vars['logPmodel'])
  return opt

def forward(opt):
    def loss_fn(ds):
        opt_local = d_of_r(opt, ds)
        opt_local = _logPmodel(opt_local)
        opt_local = compute_loss(opt_local)
        return opt_local.vars['loss']
    return loss_fn

def apply_hessian(loss_fn,ds,gradient):
    hessian_fn = jacfwd(grad(loss_fn))
    H = hessian_fn(ds)
    H += (1E-4) * jnp.eye(H.shape[0])
    return jnp.linalg.solve(H, gradient)

def clip_grad(ds,gradient):
    return jnp.clip(gradient,
                    min = -jnp.array(ds * 5./100, dtype=jnp.float64),
                    max = jnp.array(ds * 5./100, dtype=jnp.float64),
                    )

@jax.jit
def _optimization_step(idx,opt):

  loss_fn = forward(opt)
  loss_and_grad_fn = value_and_grad(loss_fn)
  loss_val, opt.vars['grad'] = loss_and_grad_fn(opt.vars['ds'])

  opt.vars['grad'] = lax.cond(pred=idx<opt.params['Nsteps']//2,
                    true_fun=lambda : clip_grad(opt.vars['ds'],opt.vars['grad']),
                    false_fun=lambda : apply_hessian(loss_fn,opt.vars['ds'],opt.vars['grad']),
                    )

  opt.vars['ds'] -= opt.vars['step'] * opt.vars['grad']  
  opt = step_schedule(idx,opt)

  # (optional) monitoring optimization:
  opt.vars['logKLs'] = opt.vars['logKLs'].at[idx].set(jnp.log(loss_val+opt.params['Pemp'] @ jnp.log(opt.params['Pemp'])))
  opt.vars['ds_optimization']= opt.vars['ds_optimization'].at[idx].set(opt.vars['ds'])

  # early stopping:
  pred = abs_relative_change(opt.vars['ds'][0],
                             opt.vars['ds_optimization'][(idx-1) % opt.params['Nsteps'],0])
  pred = pred < opt.params['threshold_early_stopping']
  opt = lax.cond(pred=pred,
                    true_fun=counter_early_stopping,
                    false_fun=do_nothing,
                    operand=opt,
                    )
  opt.vars['final_optimization_step'] += 1
  return opt

def counter_early_stopping(opt):
  opt.vars['counter_early_stopping'] += 1
  return opt

def abs_relative_change(x,y):
  return jnp.abs(y/x - 1)

def replace_nans_with_ones(float_64_tensor):
    return jnp.where(jnp.isnan(float_64_tensor), jnp.double(1.0), float_64_tensor)

def step_schedule(idx,opt):
  opt.vars['step'] = opt.params['step_initial'] + jnp.double(idx) / opt.params["Nsteps"] * (opt.params["step_final"]-opt.params["step_initial"])
  return opt

def do_nothing(opt):
  return opt

def do_nothing_idx(idx,opt):
  return opt

@jax.jit
def minimize_loss(opt):

  opt = lax.fori_loop(lower=0,
                          upper=opt.params['Nsteps'],
                          body_fun=early_stopping_optimization_step,
                          init_val=opt)
  opt.vars['logKL'] = opt.vars['logKLs'][opt.vars['final_optimization_step']]
  
  opt = d_of_r(opt, opt.vars['ds'])
  opt = _logPmodel(opt)
  return opt

@jax.jit
def early_stopping_optimization_step(idx,opt):
  opt.vars['early_stopping_condition'] = opt.vars['counter_early_stopping'] >= opt.params['tolerance_early_stopping']
  return lax.cond(opt.vars['early_stopping_condition'],
                      lambda _ : do_nothing_idx(*_),
                      lambda _ :_optimization_step(*_),
                      (idx,opt))

  
@jax.jit
def compute_KL(opt):
  opt = d_of_r(opt,opt.vars['ds'])
  opt = _logPmodel(opt)
  KL = jnp.dot(opt.params['Pemp'],
        jnp.log(opt.params['Pemp']) - jnp.log(opt.vars['Pmodel'])
  )
  opt.vars['logKL'] = jnp.log(KL)
  return opt




