Typical usage of the package
============================

A typical usage of DADApy involves the initialisation of a Data object either with a set of coordinates or with a set of
distances between points.
After the initialisation a series of computations are performed by calling the class method relative to specific
algorithm wanted.
The results of the computations are typically available as attributes of the object.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from dadapy.data import Data

    # Generate a simple 3D gaussian dataset
    X = np.random.normal(0, 1, (1000, 3))

    # initialise the "Data" class with a
    # set of coordinates
    data = Data(X)

    # compute distances up to the 100th
    # nearest neighbour
    data.compute_distances(maxk = 100)

    # compute the intrinsic dimension
    # using the 2NN estimator
    intrinsic_dim, intrinsic_dim_err = data.compute_id_2NN()

    # check the value of the intrinsic
    # dimension found
    print(data.intrinsic_dim)

    # compute the density of all points
    # using a simple kNN estimator
    log_den, log_den_err = data.compute_density_kNN(k = 15)

    # as an alternative, compute the density
    # using a more sophisticated estimator
    log_den, log_den_err = data.compute_density_PAk()

    plt.hist(data.log_den)

    # find the statistically significant peaks
    # of the density profile computed previously
    data.compute_clustering_ADP(Z = 1.5)

    print(data.N_clusters)

