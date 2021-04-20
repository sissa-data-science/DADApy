Typical usage of the package
============================

A typical usage of Duly involves the initialisation of the Data class with

.. code-block:: python

    import numpy as np
    from duly.data import Data

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

    # compute the density of all points()
    data.compute_density_kNN(k = 15)

    # find the peaks of the density profile
    data.compute_clustering()

