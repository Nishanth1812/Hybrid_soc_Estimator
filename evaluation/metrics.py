import numpy as np


def mae(y_true, y_pred):

    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def max_error(y_true, y_pred):

    return np.max(np.abs(y_true - y_pred))