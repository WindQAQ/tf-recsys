import pandas as pd

from ..utils.data_utils import get_zip_file


def load_data():
    """Loads MovieLens 1M dataset.

    Returns:
        Tuple of numpy array (x, y)
    """

    URL = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    FILE_PATH = 'ml-1m/ratings.dat'

    file = get_zip_file(URL, FILE_PATH)
    df = pd.read_csv(file, sep='::', header=None, engine='python')

    return df.iloc[:, :2].values, df.iloc[:, 2].values
