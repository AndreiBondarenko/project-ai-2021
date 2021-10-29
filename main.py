from scipy.sparse import load_npz
from mf.numba import MatrixFactorization

if __name__ == '__main__':
    x = load_npz("train_x.npz")
    y = load_npz("y_true.npz")
    x = x.transpose()
    y = y.transpose()
    M = MatrixFactorization()
    M.fit(x)
