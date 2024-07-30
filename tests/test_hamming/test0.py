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

"""Module for testing BID routines."""

import os

os.environ["JAX_ENABLE_X64"] = "True"

import pytest

from dadapy._utils.stochastic_minimization_hamming import *
from dadapy.hamming import Hamming

# EXPECTED OUTPUT
d_0 = 99.855
d_1 = 0.003
logKL = -12.39

# REPRODUCIBILITY
seed = 1
np.random.seed(seed=seed)

# DATA
L = 100  # number of bits
Ns = 5000  # number of samples
X = (
    2 * np.random.randint(low=0, high=2, size=(Ns, L)) - 1
)  # spins must be normalized to +-1

# DISTANCES
histfolder = f"./tests/test_hamming/results/hist/"
H = Hamming(coordinates=X)
H.compute_distances()
H.D_histogram(L=L, Ns=Ns, resultsfolder=histfolder)

# PARAMETER DEFINITIONS
eps = 1e-5  # good-old small epsilon
alphamin = 0  # + eps          # order of  min_quantile, to remove poorly sampled parts of the histogram
alphamax = 1  # - eps          # order of max_quantile, to define r* (named rmax here)
delta = 5e-4  # stochastic optimization step
Nsteps = int(1e6)  # number of optimization steps
seed = 1  #
optfolder0 = f"results/opt/"  # folder where optimization results are saved
export_logKLs = 1  # flag to export the logKLs during optimization

B = BID(
    H,
    alphamin=alphamin,
    alphamax=alphamax,
    seed=seed,
    delta=delta,
    Nsteps=Nsteps,
    export_logKLs=export_logKLs,
    optfolder0=optfolder0,
    L=L,
)
B.computeBID()

assert pytest.approx(B.Op.d0, abs=1e-3) == d_0
assert pytest.approx(B.Op.d1, abs=1e-3) == d_1
assert pytest.approx(jnp.log(B.Op.KL), abs=1e-2) == logKL
