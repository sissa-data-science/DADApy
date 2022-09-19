<img src="https://raw.githubusercontent.com/sissa-data-science/DADApy/master/logo/logo_1_horizontal_transparent_v2.png" width="500">

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![codecov](https://codecov.io/gh/sissa-data-science/DADApy/branch/main/graph/badge.svg?token=X4M0KWAPO5)](https://codecov.io/gh/sissa-data-science/DADApy)
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

# Generate a simple 3D gaussian dataset
X = np.random.normal(0, 1, (1000, 3))

# initialise the "Data" class with the set of coordinates
data = Data(X)

# compute distances up to the 100th nearest neighbour
data.compute_distances(maxk=100)

# compute the intrinsic dimension using 2nn estimator
data.compute_id_2NN()

# compute the density using PAk, a point adaptive kNN estimator
data.compute_density_PAk()

# find the peaks of the density profile through the ADP algorithm
data.compute_clustering_ADP()
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
  > Glielmo et al., *PNAS Nexus* (2022)


# Installation
The package is compatible with Python >= 3.7 (tested on 3.7, 3.8 and 3.9). We currently only support Unix-based systems, including Linux and macOS. 
For Windows-machines we suggest using the [Windows Subsystem for Linux (WSL)](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux).

The package requires `numpy`, `scipy` and `scikit-learn`, and `matplotlib` for the visualisations.

The package contains Cython-generated C extensions that are automatically compiled during install. 

The latest release is available through pip

```sh
pip install dadapy
```

To install the latest development version, clone the source code from github
and install it with pip as follows

```sh
git clone https://github.com/sissa-data-science/DADApy.git
cd DADApy
pip install .
```

# Citing DADApy

A description of the package is available [here](https://arxiv.org/abs/2205.03373).

Please consider citing it if you found this package useful for your research

```bib
@article{dadapy,
    author = {Glielmo, Aldo and Macocco, Iuri and Doimo, Diego and Carli, Matteo and Zeni, Claudio and Wild, Romina and d'Errico, Maria and Rodriguez, Alex and Laio Alessandro},
    title = {{DADApy: Distance-based Analysis of DAta-manifolds in Python}},
    journal = {arXiv preprint arXiv:2205.03373},
    year = {2022},
    doi = {10.48550/arXiv.2205.03373},
    url = {https://arxiv.org/abs/2205.03373},
    }
```
