import numpy as np
from tqdm.auto import tqdm
from recommenders.mfi import utils
from recommenders.mfi.lstsq import least_squares
import pickle


class MatrixFactorizationImplicit:
    """
    Matrix factorization based collaborative filtering for implicit feedback data.

    Args:
      K: size of latent space
      regularization: regularization parameter (lambda)
      iterations: number of ALS iterations to perform
    """

    def __init__(self, K=100, regularization=0.01, iterations=10):
        self.K = K
        self.regularization = regularization
        self.iterations = iterations
        self.user_factors = None
        self.item_factors = None
        self.old_recs = None

    def train(self, x, y):
        """MFI train function.

        Args:
            x (scipy.sparse.coo_matrix): training dataset
            y (scipy.sparse.coo_matrix): test dataset
        """
        # Compute unique user and items counts
        n_users, n_items = x.shape
        # Initialize latent feature matrices
        self.user_factors = np.random.normal(scale=1. / self.K, size=(n_users, self.K))
        self.item_factors = np.random.normal(scale=1. / self.K, size=(n_items, self.K))
        # Initialize "already recommended"-set
        self.old_recs = x.tocsr().copy()

        Cui_x = utils.build_conf_mat(x, alpha=40).tocsr()
        Ciu_x = utils.build_conf_mat(x, alpha=40).T.tocsr()

        Cui_y = utils.build_conf_mat(y, alpha=40).tocsr()

        train_loss = []
        test_loss = []

        # __x = x.copy()
        # __x.data = np.ones_like(__x.data)
        # __x = __x.tocsc()
        # self.popularity_scores = np.squeeze(np.asarray(__x.sum(axis=0)))
        # self.popularity_scores = self.popularity_scores / np.max(self.popularity_scores)
        # del __x

        with tqdm(range(self.iterations)) as _it:
            for _ in _it:
                # ALS
                least_squares(Cui_x, self.user_factors, self.item_factors, self.regularization)
                least_squares(Ciu_x, self.item_factors, self.user_factors, self.regularization)
                # Compute loss
                train_loss.append(
                    utils.calculate_loss(Cui_x, self.user_factors, self.item_factors, self.regularization)
                )
                test_loss.append(
                    utils.calculate_loss(Cui_y, self.user_factors, self.item_factors, self.regularization)
                )
                _it.set_postfix(train_loss=train_loss[-1], test_loss=test_loss[-1])

        return train_loss, test_loss

    def recommend(self, user, k):
        prefernces = utils.compute_preferences(user, self.user_factors, self.item_factors)
        ind = np.argsort(prefernces)[::-1]
        consumed = np.array(self.old_recs.getrow(user).nonzero()[1])
        ind = np.setdiff1d(ind, consumed, assume_unique=True)
        return ind[:k]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            m = pickle.load(f)
            self.iterations = m.iterations
            self.K = m.K
            self.regularization = m.regularization
            self.iterations = m.iterations
            self.user_factors = m.user_factors
            self.item_factors = m.item_factors
            self.old_recs = m.old_recs

