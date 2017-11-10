"""Operations related to evaluating predictions.
"""

import numpy as np


def mse(y, y_pred):
    """Returns the mean squared error between
    ground truths and predictions.
    """
    return np.mean((y - y_pred) ** 2)


def rmse(y, y_pred):
    """Returns the root mean squared error between
    ground truths and predictions.
    """
    return np.sqrt(mse(y, y_pred))


def mae(y, y_pred):
    """Returns mean absolute error between
    ground truths and predictions.
    """
    return np.mean(np.fabs(y - y_pred))
