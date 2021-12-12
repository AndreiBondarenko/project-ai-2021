import numpy as np
import scipy.sparse as sps
from tqdm.auto import tqdm
from numba import njit, prange
from recommenders.mfi import utils
import sys


@njit
def nonzeros(u: int, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray):
    """
    Returns nonzero column indices and values for row u in the sparse matrix
    represented by (rows, cols, vals) (~ scipy.sparse.coo_matrix)
    """
    i = np.searchsorted(rows, u, side='left')  # implemented as binary search
    j = np.searchsorted(rows, u, side='right')
    return cols[i:j], vals[i:j]


@njit
def build_lin_equation(Y, YtY, u, Cui_rows, Cui_cols, Cui_vals, regularization, n_factors):
    """
    Builds matrices representing linear equations to solve for ALS.
    """
    A = YtY + regularization * np.eye(n_factors)
    b = np.zeros(n_factors)
    
    nz_cols, nz_vals = nonzeros(u, Cui_rows, Cui_cols, Cui_vals)
    for _i in range(nz_cols.size):
        i = nz_cols[_i]
        confidence = nz_vals[_i]
        factor = Y[i]

        # need to make change here I think wrt. preferences
        if confidence > 0:
            b += confidence * factor
        else:
            confidence *= -1

        A += (confidence - 1) * np.outer(factor, factor)
    return A, b


@njit
def user_factor(Y, YtY, u, Cui_rows, Cui_cols, Cui_vals, regularization, n_factors):
    # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
    A, b = build_lin_equation(Y, YtY, u, Cui_rows, Cui_cols, Cui_vals, regularization, n_factors)
    return np.linalg.solve(A, b)


@njit(parallel=True)
def least_squares(Cui_rows, Cui_cols, Cui_vals, X, Y, regularization):
    
    users, n_factors = X.shape
    YtY = Y.T.dot(Y)

    for u in prange(users):
        X[u] = user_factor(Y, YtY, u, Cui_rows, Cui_cols, Cui_vals, regularization, n_factors)

if __name__ == '__main__':
    import scipy.sparse as sps
    import logging
    
    LEVEL = logging.DEBUG
    FORMAT = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
    logger = logging.getLogger('als_py')

    logger.setLevel(LEVEL)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(LEVEL)
    handler.setFormatter(FORMAT)
    logger.addHandler(handler)

    logger.debug('Loading data.')
    train = sps.load_npz('./data/train1.npz')
    test = sps.load_npz('./data/test1.npz')
    n_users, n_items = train.shape

    logger.debug('Building confidence and preference matrices.')
    Cui_sparse = utils.build_conf_mat(train)
    Pui_sparse = utils.build_pref_mat(train)

    assert utils.is_sorted(Cui_sparse.row)
    assert utils.is_sorted(Pui_sparse.row)

    logger.debug('Initializing recommender internals.')
    K = 10
    user_factors = np.random.normal(scale=1./K, size=(n_users, K))
    item_factors = np.random.normal(scale=1./K, size=(n_items, K))

    logger.debug('Computing least squares for user factors. (START)')
    least_squares(Cui_sparse.row, Cui_sparse.col, Cui_sparse.data, user_factors, item_factors, regularization=0.01)
    logger.debug('Computing least squares for user factors. (END)')