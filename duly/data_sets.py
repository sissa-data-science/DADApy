
from duly.data import Data


class Data_sets:

    def __init__(self, coordinates_list=(), distances_list=(), labels_list=(),
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

        assert (len(labels_list) == 0 or len(labels_list) == self.Nsets)

        self.data_sets = []

        for i in range(self.Nsets):
            X = coordinates_list[i]
            dists = distances_list[i]
            maxk = maxk_list[i]
            if len(labels_list) == 0:
                data = Data(coordinates=X, distances=dists, maxk=maxk, verbose=verbose, njobs=njobs)
            else:
                labels = labels_list[i]
                data = Data_wlabel(coordinates=X, distances=dists, labels=labels, maxk=maxk,
                                   verbose=verbose, njobs=njobs)

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

        if labels is None:
            data = Data(coordinates=coordinates, distances=distances, maxk=maxk,
                        verbose=self.verbose, njobs=self.njobs)
        else:
            data = Data_wlabel(coordinates=coordinates, distances=distances, labels=labels,
                               maxk=maxk, verbose=self.verbose, njobs=self.njobs)

        self.data_sets.append(data)
        self.Nsets += 1

    def set_common_gt_labels(self, labels):

        for d in self.data_sets:
            assert d.gt_labels is not None

        for d in self.data_sets:
            d.gt_labels = labels




if __name__ == '__main__':
    pass