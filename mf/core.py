import numpy as np
from tqdm.auto import tqdm
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
    def __init__(self, K=100, alpha=0.001, beta=0.02, iterations=1000):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.P = None
        self.Q = None
        self.bu = None
        self.bi = None
        self.b = None
        self.old_recs = None

    def train(self, x, y, leave_pbar=True):
        # Compute unique user and items counts
        num_users, num_items = x.shape
        # Initialize latent feature matrices
        self.P = np.random.normal(scale=1./self.K, size=(num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(num_items, self.K))
        # Initialize biases
        self.bu = np.zeros(num_users)
        self.bi = np.zeros(num_items)
        self.b = np.sum(x.data) / x.count_nonzero()
        # Initialize "already recommended"-set
        self.old_recs = {}
        for i, j, _ in zip(x.row, x.col, x.data):
            self.old_recs.setdefault(i, set()).add(j)

        _x = np.array(list(zip(x.row, x.col, x.data)))
        _y = np.array(list(zip(y.row, y.col, y.data)))

        train_sse = []
        test_sse = []

        with tqdm(range(self.iterations), leave=leave_pbar) as _it:
            for _ in _it:
                fast.sgd(_x, self.P, self.Q, self.bu, self.bi, self.b, self.alpha, self.beta)
                train_error = fast.sse(_x, self.P, self.Q, self.bu, self.bi, self.b)
                test_error = fast.sse(_y, self.P, self.Q, self.bu, self.bi, self.b)
                train_sse.append(train_error)
                test_sse.append(test_error)
                _it.set_postfix(test_error=test_error, train_error=train_error)
        return train_sse, test_sse

    def recommend(self, k: int, user: int):
        scores = fast.compute_relevance_scores(user, self.P, self.Q, self.bu, self.bi, self.b)
        # sort items in descending order by their score
        ind = np.argsort(scores)[::-1]
        # remove already recommended items, i.e. items from training set
        ind = ind[np.where(ind not in self.old_recs[user])]
        # take top-k
        return ind[:k]
