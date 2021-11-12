import numpy as np
import scipy.sparse as sps


class Popularity:
    def __init__(self):
        self.scores = None
        self.ranking = None

    def train(self, x: sps.coo_matrix):
        _x = x.copy()
        _x.data = np.ones_like(_x.data)
        _x = _x.tocsc()

        self.scores = np.squeeze(np.asarray(_x.sum(axis=1)))
        self.ranking = np.argsort(self.scores)[::-1]

    def recommend(self, user: int, k: int):
        """
        Make a top-k recommendation for user
        :param user: id of user to make recommendation for
        :param k: number of items to recommend
        :return: top-k recommended item id's
        """
        return self.ranking[:k]