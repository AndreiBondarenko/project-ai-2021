import numpy as np
from tqdm.auto import tqdm
from mf import fast
import pickle

class MatrixFactorization:
    """
    Matrix factorization based collaborative filtering.

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
        """MF train function.

        Args:
            x (scipy.sparse.coo_matrix): training dataset
            y (scipy.sparse.coo_matrix): test dataset (sparse coo matrix)
            leave_pbar (bool): toggle whether to leave progress bar after finished.


        Returns:
            The return value. True for success, False otherwise.

        """

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
        self.old_recs = x.tocsr()

        _x = np.array(list(zip(x.row, x.col, x.data)))
        _y = np.array(list(zip(y.row, y.col, y.data)))

        train_mse = []
        test_mse = []

        with tqdm(range(self.iterations), leave=leave_pbar) as _it:
            for _ in _it:
                fast.sgd(_x, self.P, self.Q, self.bu, self.bi, self.b, self.alpha, self.beta)
                train_error = fast.mse(_x, self.P, self.Q, self.bu, self.bi, self.b)
                test_error = fast.mse(_y, self.P, self.Q, self.bu, self.bi, self.b)
                train_mse.append(train_error)
                test_mse.append(test_error)
                _it.set_postfix(test_error=test_error, train_error=train_error)
        return train_mse, test_mse


    def recommend(self, user, k):
        scores = fast.compute_relevance_scores(user, self.P, self.Q, self.bu, self.bi, self.b)
        # sort items in descending order by their score
        ind = np.argsort(scores)[::-1]
        # remove consumed items, i.e. items from training set
        consumed = np.array(self.old_recs.getrow(user).nonzero()[1])
        ind = np.setdiff1d(ind, consumed, assume_unique=True)
        # take top-k
        return ind[:k]


    def recommend_sim(self, user, k):
        # self.old_recs --> scipy.sparse.csr_matrix
        history = self.old_recs.getrow(user)
        # get highest rated item in user's history
        best = history.argmax()
        # compute similarity scores between highest rated item
        # and all other items
        scores = fast.compute_item_similarities(best, self.Q)
        # sort items based on similarity scores
        ind = np.argsort(scores)[::-1]
        # remove already consumed items
        consumed = np.array(history.nonzero()[1])
        ind = np.setdiff1d(ind, consumed, assume_unique=True)
        # return top k
        return ind[:k]

 
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


    def load(self, path):
        with open(path, 'rb') as f:
            m = pickle.load(f)
            self.K = m.K
            self.alpha = m.alpha
            self.beta = m.beta
            self.iterations = m.iterations
            self.P = m.P
            self.Q = m.Q
            self.bu = m.bu
            self.bi = m.bi
            self.b = m.b
            self.old_recs = m.old_recs

