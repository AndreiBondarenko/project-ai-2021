from scipy.sparse import load_npz
from matrix_factorization.core import MatrixFactorization

import json
import gzip
from tqdm import tqdm
import pandas as pd

def chunked_read_json_fast(filename, chunksize=100_000, nrows=None):
    """
    Loads line delimited JSON files in chunks of chunksize.
    This prevents out-of-memory issues for large datasets.
    If this function still gives you out-of-memory issues,
    update the pandas library to a more recent version.
    
    Args:
        filename: path of the JSON file
        chunksize: number of rows to read from the file at a time
        nrows: total number of rows to read from the file
  
    Returns:
        Pandas DataFrame containing (part of) the data. 
    """
    df_lst = []
    chunk_data = []
    lines = 0
    if nrows and nrows <= 0:
        return None
    with gzip.open(interactions_file, 'rb') as f:
        for line in tqdm(f):
            chunk_data.append(json.loads(line.decode().strip('\n')))
            lines += 1
            if (len(chunk_data) >= chunksize) or (nrows and (lines >= nrows)):
                df_lst.append(pd.DataFrame.from_records(chunk_data))
                chunk_data.clear()
            if nrows and (lines >= nrows):
                break;

    df = pd.concat(df_lst, ignore_index=True)
    return df


if __name__ == '__main__':
    x = load_npz("train_x.npz")
    y = load_npz("y_true.npz")
    x = x.transpose()
    y = y.transpose()
    M = MatrixFactorization()
    M.fit(x)
