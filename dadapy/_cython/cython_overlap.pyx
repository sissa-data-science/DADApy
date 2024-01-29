# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import cython
import numpy as np
cimport numpy as np

cimport numpy as np
from libc.math cimport exp
from libc.stdint cimport int32_t

DTYPE = np.int64
floatTYPE = np.float64

ctypedef np.int_t DTYPE_t
ctypedef np.float64_t floatTYPE_t

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def _compute_data_overlap(DTYPE_t Nele,
                    DTYPE_t k,
                    np.ndarray[DTYPE_t, ndim = 2] dist_indices1,
                    np.ndarray[DTYPE_t, ndim = 2] dist_indices2,
):

    cdef np.ndarray[floatTYPE_t, ndim=1] overlaps = -np.ones(Nele)
    cdef floatTYPE_t count
    cdef DTYPE_t i, j, l
    cdef DTYPE_t[:, ::1] indices1 = dist_indices1
    cdef DTYPE_t[:, ::1] indices2 = dist_indices2

    for i in range(Nele):
        count = 0
        for j in range(1, k+1):
            elem = indices1[i, j]
            for l in range(1, k + 1):
                if elem == indices2[i, l]:
                    count += 1
                    break
        overlaps[i] = count / k

    return overlaps



def return_label_overlap(self, int[:] labels, int k=30, bint avg=True, coords=None, float k_per_classes=0.1):
    """Return the neighbour overlap between the full space and a set of labels.

    An overlap of 1 means that all neighbours of a point have the same label as the central point.

    Args:
        labels (list): the labels with respect to which the overlap is computed
        k (int): the number of neighbours considered for the overlap

    Returns:
        (float): the neighbour overlap of the points
    """
    cdef:
        Py_ssize_t num_labels = labels.shape[0]
        int[:] class_count
        bint class_imbalance
        int max_class_count
        int[:] k_per_classes_arr
        Py_ssize_t i
        int[:, :] neighbor_index
        int[:, :] ground_truth_labels
        bint[:, :] overlaps
        int cols, col
        float overlap_sum = 0
        float total_k_classes = 0
        float overlaps_result = 0

    class_count = np.bincount(labels)

    class_imbalance = not np.all(class_count == np.repeat(class_count[0], class_count.shape[0]))
    if class_imbalance:
        max_class_count = np.max(class_count)
        k = int(max_class_count * k_per_classes)

    dist_indices, k = self._get_nn_indices(self.X, self.distances, self.dist_indices, k, coords)
    assert num_labels == dist_indices.shape[0]

    neighbor_index = self.dist_indices[:, 1:k+1]
    ground_truth_labels = np.repeat(np.array([labels]).T, repeats=k, axis=1)
    overlaps = np.equal(np.array(labels)[neighbor_index], ground_truth_labels)

    if class_imbalance:
        k_per_classes_arr = np.array([int(class_count[labels[i]] * k_per_classes) for i in range(num_labels)])
        if avg:
            cols = overlaps.shape[1]
            for i in range(num_labels):
                for col in range(cols):
                    if col >= k_per_classes_arr[i]:
                        overlaps[i, col] = False
                overlap_sum += np.sum(overlaps[i, :])
                total_k_classes += k_per_classes_arr[i]
            overlaps_result = overlap_sum / total_k_classes
            return overlaps_result
        else:
            for i in range(num_labels):
                overlaps[i, :] = np.sum(overlaps[i, :]) / k_per_classes_arr[i]
                return overlaps
    else:
        if avg:
            overlaps_result = np.mean(overlaps)
            return overlaps_result
        else:
            overlaps = np.mean(overlaps, axis=1)
            return overlaps

    