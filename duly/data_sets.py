import numpy as np
from duly.data import Data


class Data_sets:

    def __init__(self, coordinates_list=(), distances_list=(),
                 maxk_list=[None], verbose=False, njobs=1):
        if len(distances_list) == 0:
            self.Nsets = len(coordinates_list)
            distances_list = [None] * self.Nsets

        elif len(coordinates_list) == 0:
            self.Nsets = len(distances_list)
            coordinates_list = [None] * self.Nsets

        else:
            assert len(coordinates_list) == len(distances_list)
            self.Nsets = len(coordinates_list)

        if len(maxk_list) == 1:
            maxk_list = [maxk_list[0]] * self.Nsets

        self.data_sets = []

        for i in range(self.Nsets):
            X = coordinates_list[i]
            dists = distances_list[i]
            maxk = maxk_list[i]
            data = Data(coordinates=X, distances=dists, maxk=maxk, verbose=verbose, njobs=njobs)

            self.data_sets.append(data)

        # self.maxk = min([self.data_sets[i].maxk for i in range(self.Nsets)])  # maxk neighbourhoods

        self.verbose = verbose
        self.njobs = njobs
        self.ids = None  # ids
        self.ov_gt = None  # overlap ground truth (classes)
        self.ov_out = None  # overlap output neighborhoods
        self.ov_ll = None  # overlap ll neighbourhoods
        self.gamma = None  # gamma_matrix all to all

    def add_one_dataset(self, coordinates=None, distances=None, labels=None, maxk=None):

        data = Data(coordinates=coordinates, distances=distances, maxk=maxk,
                    verbose=self.verbose, njobs=self.njobs)

        self.data_sets.append(data)
        self.Nsets += 1

    def set_common_gt_labels(self, labels):

        for d in self.data_sets:
            assert d.gt_labels is not None

        for d in self.data_sets:
            d.gt_labels = labels

    def compute_id_2NN(self, decimation=1, fraction=0.9, n_reps=1):

        print('computing id: fraction = {}, decimation = {}, repetitions = {}, range = {}'.format(fraction, decimation,
                                                                                                  n_reps, 2))
        for i, d in enumerate(self.data_sets):
            print('computing id of dataset ', i)
            d.compute_id_2NN(decimation=decimation, fraction=fraction)
            print('id computation finished')

        self.ids = [d.id_selected for d in self.data_sets]

    def compute_id_scaling(self, range_max=1024, d0=0.001, d1=1000, return_ids=False, save_mus=False):

        for i, d in enumerate(self.data_sets):
            print('computing id of layer ', i)
            d.compute_id_scaling(range_max=range_max, d0=d0, d1=d1, return_ids=return_ids, save_mus=save_mus)

        self.ids = [d.ids_scaling[0] for d in self.data_sets]

    def serialize_computation(self, computation_string, **kwargs):
        """This method applies the computation defined in 'computation_string' to all Data instances
        contained in the class.
        """
        for i, d in enumerate(self.data_sets):
            comput = getattr(d, computation_string)
            comput(**kwargs)


if __name__ == '__main__':
    X1 = np.random.uniform(0, 1, (100, 2))
    X2 = np.random.uniform(0, 1, (100, 2))

    ds = Data_sets(coordinates_list=([X1, X2]))

    ds.serialize_computation('compute_distances', maxk=3)

    print(ds.data_sets[0].distances)
