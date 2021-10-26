import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def sgd(x, P, Q, bu, bi, b, alpha, beta):
    for idx in prange(len(x)):
        i, j, v = x[idx]
        eij = v - predict_rating(i, j, P, Q, bu, bi, b)
        # Retrieve pre-update values
        Pi = P[i, :].copy()
        Qj = Q[:, j].copy()
        # Update latent features
        P[i, :] += alpha * (eij * Qj - beta * Pi)
        Q[:, j] += alpha * (eij * Pi - beta * Qj)
        # Update biases
        bu[i] += alpha * (eij - beta * bu[i])
        bi[j] += alpha * (eij - beta * bi[j])


@jit(nopython=True)
def frob(x, P, Q, bu, bi, b):
    error = 0
    for i, j, v in x:
        abs_error = v - predict_rating(i, j, P, Q, bu, bi, b)
        error += abs_error ** 2
    return np.sqrt(error)


@jit(nopython=True)
def predict_rating(i, j, P, Q, bu, bi, b):
    return b + bu[i] + bi[j] + np.dot(P[i, :], Q[:, j])

