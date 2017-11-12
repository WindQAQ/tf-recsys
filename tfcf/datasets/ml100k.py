import pandas as pd

from ..utils.data_utils import get_zip_file


def load_data():
    """Loads MovieLens 100k dataset.

    Returns:
        Tuple of numpy array (x, y)
    """

    URL = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    FILE_PATH = 'ml-100k/u.data'

    file = get_zip_file(URL, FILE_PATH)
    df = pd.read_csv(file, sep='\t', header=None)

    return df.iloc[:, :2].values, df.iloc[:, 2].values
