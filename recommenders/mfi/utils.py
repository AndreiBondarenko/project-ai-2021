import numpy as np
import scipy.sparse as sps
from typing import Callable
from numba import njit


def build_conf_mat(x: sps.coo_matrix, alpha=40):

    def rat_to_conf(r):
        if r == 0:
            return 1
        elif r < 2.5:
            return alpha * (5-r)
        else:
            return alpha * r
    # Sparse representation! Doesn't contain 1 values
    # for unobserved user-item pairs and all present
    # values should be incremented by 1 to get the
    # actual confidence value
    return sparse_apply_elementwise(x, rat_to_conf)


def build_pref_mat(x: sps.coo_matrix):
    return sparse_apply_elementwise(x, lambda e: 1 if e > 2.5 else 0)


def sparse_apply_elementwise(x: sps.coo_matrix, f: Callable):
    """
    Applies function f elementwise to the elements of sparse matrix x.
    Additionally, prunes any indices with value equal to 0.
    Args:
        x: sparse matrix
        f: function to apply to non-null elements

    Returns:
        Sparse matrix resulting from applying f to
        non-null elements of x.
    """
    _x = x.copy()
    _x.data[:] = np.vectorize(f)(_x.data)
    # prune zeros
    zeros = np.nonzero(_x.data == 0)
    _x.row = np.delete(_x.row, zeros)
    _x.col = np.delete(_x.col, zeros)
    _x.data = np.delete(_x.data, zeros)
    return _x

@njit
def is_sorted(a: np.ndarray):
    for i in range(a.size-1):
        if a[i+1] < a[i] :
            return False
    return True
