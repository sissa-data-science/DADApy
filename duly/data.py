import multiprocessing
import os

import numpy as np
from duly.metric_comparisons import MetricComparisons
from duly.clustering import Clustering

cores = multiprocessing.cpu_count()

np.set_printoptions(precision=2)

os.getcwd()


class Data(Clustering, MetricComparisons):
    """This class is a container of all the method of DULY. It is initialised with a set of coordinates or a set of
    distances, and all methods can be called on the generated class instance.

    Attributes:

    """
    def __init__(self, coordinates=None, distances=None, maxk=None, verbose=True, njobs=cores, working_memory=1024):

        super().__init__(coordinates=coordinates, distances=distances, maxk=maxk, verbose=verbose,
                         njobs=njobs)

