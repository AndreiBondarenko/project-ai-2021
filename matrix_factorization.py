import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


class MatrixFactorization:
    """
    Matrix factorization based collaborative filtering .

    Args:
      K: number of latent dimensions
      alpha: SGD learning rate
      beta: regularization parameter
      iterations: number of SGD iterations to perform
    """
    def __init__(self, K=100, alpha=0.1, beta=0.01, _lambda=0.3, iterations=100):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self._lambda = _lambda
        self.iterations = iterations
        self.P = None
        self.Q = None
        self.bu = None
        self.bi = None
        self.b = None

    def fit(self, x):
        """

        Args:
          x: user-item rating matrix (sparse)
        """
        num_users, num_items = x.shape
        # Initialize latent feature matrices
        self.P = np.random.normal(scale=1./self.K, size=(num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.K, num_items))
        # Initialize biases
        self.bu = np.zeros(num_users)
        self.bi = np.zeros(num_items)
        self.b = x.sum() / len(x.nonzero()[0])

        _x = x.todok()  # to python dict

        errors = []

        with tqdm(range(self.iterations)) as _it:
            for _ in _it:
                self.sgd(_x)
                error = self.frobenius(_x)
                errors.append(error)
                _it.set_postfix(error=error)

        plt.plot(errors)
        plt.show()

    def sgd(self, x):
        for (i, j), v in x.items():
            eij = v - self.predict_rating(i, j)
            # Retrieve pre-update values
            Pi = self.P[i, :].copy()
            Qj = self.Q[:, j].copy()
            # Update latent features
            self.P[i, :] += self.alpha * (eij * Qj - self.beta * Pi)
            self.Q[:, j] += self.alpha * (eij * Pi - self.beta * Qj)
            # Update biases
            self.bu[i] += self.alpha * (eij - self.beta * self.bu[i])
            self.bi[j] += self.alpha * (eij - self.beta * self.bi[j])

    def frobenius(self, x):
        error = 0
        for (i, j), v in x.items():
            abs_error = v - self.predict_rating(i, j)
            error += abs_error ** 2
        return np.sqrt(error)

    def predict_rating(self, i, j):
        return self.b + self.bu[i] + self.bi[j] + np.dot(self.P[i, :], self.Q[:, j])

    def predict(self, x):
        _x = x.todok()  # to python dict
        pass
