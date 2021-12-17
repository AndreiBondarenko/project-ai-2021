import numpy as np
import scipy.sparse as sps
from typing import Callable
from numba import njit, prange


def build_conf_mat(x: sps.coo_matrix, alpha=40):

    def rat_to_conf(r):
        if r == 0:
            return 1
        elif r < 2.5:
            return - alpha * (5-r)
        else:
            return alpha * r
    # Sparse representation! Doesn't contain 1 values
    # for unobserved user-item pairs and all present
    # values should be incremented by 1 to get the
    # actual confidence value
    return sparse_apply_elementwise(x, rat_to_conf)


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


def calculate_loss(Cui, X, Y, regularization):
    return fast_calculate_loss(Cui.indptr, Cui.indices, Cui.data, X, Y, regularization)


@njit(parallel=True)
def fast_calculate_loss(Cui_indptr, Cui_indices, Cui_data, X, Y, regularization):
    users = X.shape[0]
    items = Y.shape[0]
    user_loss = np.zeros(users)
    user_confidence = np.zeros(users)
    user_norm_entries = np.empty(users)
    item_norm = 0

    YtY = np.dot(np.transpose(Y), Y)
    # Calculate loss = SUM(u,i)[Cui(u,i)(Pui(u,i) - X(u)Y(i))^2] + regularization*(user_norm^2 + item_norm^2)
    for u in prange(users):
        # calculate r = YtY * Xu
        r = np.dot(np.triu(YtY, k=0), X[u])
        for index in range(Cui_indptr[u], Cui_indptr[u + 1]):
            i = Cui_indices[index]
            confidence = Cui_data[index]

            if confidence > 0:
                temp = -2 * confidence
            else:
                temp = 0
                confidence = -1 * confidence

            # calculates (-2 * confidence) + (confidence - 1) * YiXu
            temp = temp + (confidence - 1) * np.dot(Y[i], X[u])
            # calculates r = [(-2 * confidence) + (confidence - 1) * YiXu]Yi + YtY*Xu
            r = temp * Y[i] + r
            user_confidence[u] += confidence
            user_loss[u] += confidence

        # calculates [[(-2 * confidence) + (confidence - 1) * YiXu]Yi + YtY*Xu] * Xu
        # = [(-2 * confidence) + (confidence - 1) * YiXu]YiXu + YtY*XuXu
        # = -2*confidence*YiXu + confidence*YtYXuXu
        user_loss[u] += np.dot(r, X[u])
        user_norm_entries[u] = np.dot(X[u], X[u])

    for i in range(items):
        item_norm += np.dot(Y[i], Y[i])

    user_norm = user_norm_entries.sum()
    loss = user_loss.sum()
    loss += regularization * (item_norm + user_norm)
    total_confidence = user_confidence.sum()
    return loss / (total_confidence + users * items - Cui_data.size)


@njit(parallel=True)
def compute_preferences(u, user_factors, item_factors):
    scores = np.empty(len(item_factors))
    for i in prange(len(item_factors)):
        scores[i] = np.dot(user_factors[u], item_factors[i])
    return scores
