import numpy as np
from numba import jit, prange

# Do no paralellize sgd to avoid calculating gradients with stale copies of P and Q
@jit(nopython=True)
def sgd(x, P, Q, bu, bi, b, alpha, beta):
    for idx in prange(len(x)):
        i, j, v = x[idx]
        eij = v - predict_rating(i, j, P, Q, bu, bi, b)
        # Retrieve pre-update values
        Pi = P[i].copy()
        Qj = Q[j].copy()
        # Update latent features
        P[i] += alpha * (eij * Qj - beta * Pi)
        Q[j] += alpha * (eij * Pi - beta * Qj)
        # Update biases
        bu[i] += alpha * (eij - beta * bu[i])
        bi[j] += alpha * (eij - beta * bi[j])


@jit(nopython=True, parallel=True, cache=True)
def mse(x, P, Q, bu, bi, b):
    error = 0
    for idx in prange(len(x)):
        i, j, v = x[idx]
        abs_error = v - predict_rating(i, j, P, Q, bu, bi, b)
        error += abs_error ** 2
    return error / len(x)


@jit(nopython=True, cache=True)
def predict_rating(i, j, P, Q, bu, bi, b):
    return b + bu[i] + bi[j] + np.dot(P[i], Q[j])


@jit(nopython=True, parallel=True, cache=True)
def compute_relevance_scores(i, P, Q, bu, bi, b):
    scores = np.empty(len(Q))
    for j in prange(len(Q)):
        scores[j] = b + bu[i] + bi[j] + np.dot(P[i], Q[j])
    return scores


def compute_item_similarities(i, Q):
    As = Q[i]
    dot = As @ Q.T
    norm_a = np.linalg.norm(As, axis=1)
    norm_b = np.linalg.norm(Q, axis=1)
    sim = np.divide(np.divide(dot.T, norm_a).T, norm_b)
    return np.sum(sim, axis=0) / len(i)
