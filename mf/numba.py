import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mf import fast


class MatrixFactorization:
    """
    Matrix factorization based collaborative filtering .

    Args:
      K: number of latent dimensions
      alpha: SGD learning rate
      beta: regularization parameter
      iterations: number of SGD iterations to perform
    """
    def __init__(self, K=100, alpha=0.1, beta=0.01, iterations=1000):
        self.K = K
        self.alpha = alpha
        self.beta = beta
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
        self.Q = np.random.normal(scale=1./self.K, size=(num_items, self.K))
        # Initialize biases
        self.bu = np.zeros(num_users)
        self.bi = np.zeros(num_items)
        self.b = x.sum() / len(x.nonzero()[0])

        _x = np.array([(i, j, v) for (i, j), v in x.todok().items()])  # to python dict

        errors = []

        with tqdm(range(self.iterations)) as _it:
            for _ in _it:
                fast.sgd(_x, self.P, self.Q, self.bu, self.bi, self.b, self.alpha, self.beta)
                error = fast.frob(_x, self.P, self.Q, self.bu, self.bi, self.b)
                errors.append(error)
                _it.set_postfix(error=error)

        plt.plot(errors)
        plt.show()

    def predict(self, x):
        pass
