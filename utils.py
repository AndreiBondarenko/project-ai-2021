from tqdm.auto import tqdm
import pandas as pd
import itertools
import json
import gzip
import numpy as np
import scipy.sparse as sps
from collections import defaultdict


def read_json_fast(filename, nrows=None):
    """
    Loads line delimited JSON files faster than the
    read_json function provided by Pandas.

    Iterates over file line per line, so shouldn't
    cause out-of-memory issues, except if resulting
    DataFrame is too big.

    Args:
      filename: path of the JSON file
      nrows: total number of rows to read from the file

    Returns:
      Pandas DataFrame containing (part of) the data.
    """
    with gzip.open(filename) as f:
        d = defaultdict(list)

        print(f"Processing {filename.split('/')[-1]}:")
        if nrows is not None:
            pbar = tqdm(itertools.islice(f, nrows), unit="lines")
        else:
            pbar = tqdm(f, unit="lines")
        for l in pbar:
            for k, v in json.loads(l).items():
                d[k].append(v)
        return pd.DataFrame(d)


def recall_at_k(topk: np.ndarray, actual: sps.csr_matrix):
    """
    Compute recall@k for given set of recommended items

    Args:
        topk: indices of recommended items
        actual: sparse representation the actual future items
                (use csr since column access within the same row is still efficient)

    Returns: recall@k
    """
    return actual[:, topk].count_nonzero() / actual.count_nonzero()


def precision_at_k(topk: np.ndarray, actual: sps.csr_matrix):
    """
    Compute precision@k for given set of recommended items

    Args:
        topk: indices of recommended items
        actual: sparse representation the actual future items
                (use csr since column access within the same row is still efficient)

    Returns: precision@k
    """
    return actual[:, topk].count_nonzero() / len(topk)


def ndcg_at_k(topk: np.ndarray, actual: sps.csr_matrix):
    """
    Compute ndcg@k for given set of recommended items

    Args:
        topk: indices of recommended items
        actual: sparse representation the actual future items
                (use csr since column access within the same row is still efficient)

    Returns: ndcg@k
    """
    ratings = actual.toarray()[0]
    num = ratings[topk]
    den = np.log2(np.arange(2, len(topk) + 2))
    dcg = np.divide(num, den).sum()
    if dcg == 0:
        return 0
    inum = np.sort(num)[::-1]
    idcg = np.divide(inum, den).sum()
    return dcg / idcg
