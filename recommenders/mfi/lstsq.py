import numpy as np
from numba import njit, prange
from recommenders.mfi import utils
import sys


@njit
def nonzeros(indptr, indices, data, row):
    """
    Returns non-zero indices and values of a row in
    a scipy.sparse.csr_matrix.

    Args:
        indptr: CSR format index pointer array of the matrix
        indices: CSR format index array of the matrix
        data: CSR format data array of the matrix
        row: Row for which to return indices and values

    Returns:
        (indices, data) where indices is an array containing
            the indices of non-zero elements and data contains
            the associated values.
    """
    start, stop = indptr[row], indptr[row + 1]
    return indices[start:stop], data[start:stop]


@njit
def lin_equation(Y, YtY, Cui_indptr, Cui_indices, Cui_data, u, regularization, n_factors):
    """
    Builds matrices to solve for least squares optimization.

    Args:
        Y: item factor matrix
        YtY: pre-computed Y^T.Y matrix
        Cui_indptr: CSR format index pointer array of the confidence matrix
        Cui_indices: CSR format index array of the confidence matrix
        Cui_data: CSR format data array of the confidence matrix
        u: user to build linear equation for
        regularization: regularization parameter (lambda)
        n_factors: size of the latent space

    Returns:
        (A, b) which together represent a system of linear equations
            A = YtY regularization * I + Yt(Cu-I)Y
            b = YtCuPu
    """
    A = YtY + regularization * np.eye(n_factors)
    b = np.zeros(n_factors)
    
    indices, data = nonzeros(Cui_indptr, Cui_indices, Cui_data, u)
    for _i in range(indices.size):
        i = indices[_i]
        confidence = data[_i]
        factor = Y[i]

        if confidence > 0:
            b += confidence * factor
        else:
            confidence *= -1

        A += (confidence - 1) * np.outer(factor, factor)
    return A, b


@njit(parallel=True)
def fast_least_squares(Cui_indptr, Cui_indices, Cui_data, X, Y, regularization):
    """
    Perform least squares optimization of X, keeping Y fixed.

    Args:
        Cui_indptr: CSR format index pointer array of the confidence matrix
        Cui_indices: CSR format index array of the confidence matrix
        Cui_data: CSR format data array of the confidence matrix
        X: user factor matrix
        Y: item factor matrix
        regularization: regularization parameter (lambda)

    Returns:
        Nothing
    """
    
    users, n_factors = X.shape
    YtY = Y.T.dot(Y)

    for u in prange(users):
        A, b = lin_equation(Y, YtY, Cui_indptr, Cui_indices, Cui_data, u, regularization, n_factors)
        X[u] = np.linalg.solve(A, b)


def least_squares(Cui, X, Y, regularization):
    fast_least_squares(Cui.indptr, Cui.indices, Cui.data, X, Y, regularization)


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

    logger.debug('Building confidence matrix.')
    Cui_sparse = utils.build_conf_mat(train).tocsr()

    logger.debug('Initializing recommender internals.')
    K = 100
    user_factors = np.random.normal(scale=1./K, size=(n_users, K))
    item_factors = np.random.normal(scale=1./K, size=(n_items, K))

    logger.debug('Computing least squares for user factors. (START)')
    least_squares(Cui_sparse, user_factors, item_factors, regularization=0.01)  # takes about 8 minutes with K=100
    logger.debug('Computing least squares for user factors. (END)')
