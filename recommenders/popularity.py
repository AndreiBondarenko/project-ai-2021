import numpy as np
import scipy.sparse as sps


class Popularity:
    def __init__(self):
        self.scores = None
        self.ranking = None
        self.old_recs = None

    def train(self, x: sps.coo_matrix):
        self.old_recs = x.tocsr()

        _x = x.copy()
        _x.data = np.ones_like(_x.data)
        _x = _x.tocsc()

        self.scores = np.squeeze(np.asarray(_x.sum(axis=0)))
        self.scores = self.scores / self.scores.max()
        self.ranking = np.argsort(self.scores)[::-1]

    def recommend(self, user: int, k: int):
        """
        Make a top-k recommendation for user

        Args:
            user: id of user to make recommendation for
            k: number of items to recommend

        Returns:
            top-k recommended item id's
        """
        consumed = np.array(self.old_recs.getrow(user).nonzero()[1])
        ind = np.setdiff1d(self.ranking, consumed, assume_unique=True)
        return ind[:k]
