import os
import inspect
import tensorflow as tf


def _class_vars(obj):
    return {k: v for k, v in inspect.getmembers(obj)
            if not k.startswith('__') and not callable(k)}


class BaseModel(object):
    """Base model for SVD and SVD++.
    """

    def __init__(self, config):
        self._built = False

        for attr in _class_vars(config):
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(config, attr))
