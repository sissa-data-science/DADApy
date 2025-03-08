Implemented Algorithms
======================

The algorithms currently implemented in the package can be divided in five broad groups.


Intrinsic dimension estimation
--------------------------------

These algorithms estimate the *intrinsic dimension* of the data manifold i.e., the minimum number of coordinates needed
to describe the manifold without a significant loss of information.
The algorithms currently implemented are:

* Two NN ("Two nearest neighbour estimator")
* Gride ("Generalized ratios id estimator")
* BID ("Binary intrinsic dimension")

Density estimation
-----------------------

These algorithms estimate the density profile from which the dataset was sampled.
The algorithms currently implemented are:

* k-NN ("k-nearest neighbours estimator")
* PAk ("Point adaptive k-NN estomator")
* k*-NN ("k-star nearest neighbours estimator")
* point-adaptive mean-shift gradient estimator
* BMTI ("Binless Multidimensional Thermodynamic Integration")

Density based clustering
--------------------------

These algorithms find the statistically significant peaks of the density profile and use this information to divide the
dataset into clusters of data.
The algorithms currently implemented are:

* DP ("Density peaks clustering")
* ADP("Advanced density peaks clustering")

Metric space comparison
--------------------------

These algorithms estimate and quantify whether two spaces endowed with a distance measure are equivalent or not,
and whether one space is more informative than the other.
The algorithms currently implemented are:

* Neighbourhood overlap
* Information imbalance

Feature weighting / Differentiable Information Imbalance
-----------------------------------------------------------

This algorithm estimates the information content of an input set of features with respect to a ground truth (which
can be the full set), assigning an optimal weight to each feature. The weights can take the value zero, leading to 
feature selection. The JAX implementation of this method, in the class DiffImbalance, is not compatible with Python 
versions lower than 3.9.
The algorithm currently implemented is:

* DII ("Differentiable Information Imbalance") 
