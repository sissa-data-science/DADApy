<img src="https://raw.githubusercontent.com/sissa-data-science/DADApy/master/logo/logo_1_horizontal_v2.jpg" width="200">

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/sissa-data-science/DADApy/branch/develop/graph/badge.svg?token=X4M0KWAPO5)](https://codecov.io/gh/sissa-data-science/DADApy)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/sissa-data-science/dadapy/test?label=test)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/sissa-data-science/dadapy/lint?label=lint)

DADApy is a Python package for the characterisation of manifolds in high dimensional spaces.


# Homepage
For more details and tutorials, visit the homepage at:
[https://dadapy.readthedocs.io/](https://dadapy.readthedocs.io/)

# Quick Example

```python
import numpy as np
from dadapy.data import Data

# a simple 3D gaussian dataset
X = np.random.normal(0, 1, (1000, 3))

# initialise the "Data" class with a
# set of coordinates
data = Data(X)

# compute distances up to the 100th
# nearest neighbour
data.compute_distances(maxk = 100)

# compute the intrinsic dimension
data.compute_id_2NN()

# compute the density of all points using a basic kNN estimator
data.compute_density_kNN(k = 15)

# or using a more advanced PAk estimator
data.compute_density_PAk()

# find the peaks of the density profile
data.compute_clustering()
```

# Currently implemented algorithms

- Intrinsic dimension estimators
     - 
- Two-NN estimator 
  > Facco et al., *Scientific Reports* (2017)
- Gride estimator
  > Denti et al., *arXiv* (2021)
- Density estimators
    - 
- kNN estimator
- k*NN estimator (kNN with adaptive choice of k)
- PAk estimator
  > Rodriguez et al., *JCTC* (2018)

- Density peaks estimators
    - 
- Density peaks clustering 
  > Rodriguez and Laio, *Science* (2014)
- Advanced density peaks clustering
  > d’Errico et al., *Information Sciences* (2021)
- k-peak clustering
  > Sormani, Rodriguez and Laio, *JCTC* (2020)

- Manifold comparison tools
    - 
- Neighbourhood overlap
  > Doimo et al., *NeurIPS* (2020)
- Information imbalance
  > Glielmo et al., *ArXiv* (2021)


# Installation
The package is compatible with Python >= 3.6 (tested on 3.6, 3.7, 3.8 and 3.9). We currently only support Unix-based systems, including Linux and macOS. 
For Windows-machines we suggest using the [Windows Subsystem for Linux (WSL)](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux).
The exact list of dependencies are given in setup.py and all of them will be automatically installed during setup.

The package contains Cython-generated C extensions that are automatically compiled during install. 

The latest stable release is (not yet!) available through pip: (add the `--user` flag if root access is not available

```sh
pip install dadapy
```

To install the latest development version, clone the source code from github
and install with pip from local file:

```sh
git clone https://github.com/sissa-data-science/DADApy.git
cd DADApy
pip install .
```
