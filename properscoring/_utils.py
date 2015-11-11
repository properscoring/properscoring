import warnings
import contextlib

import numpy as np


def move_axis_to_end(array, axis):
    array = np.asarray(array)
    return np.rollaxis(array, axis, start=array.ndim)


def argsort_indices(a, axis=-1):
    """Like argsort, but returns an index suitable for sorting the
    the original array even if that array is multidimensional
    """
    a = np.asarray(a)
    ind = list(np.ix_(*[np.arange(d) for d in a.shape]))
    ind[axis] = a.argsort(axis)
    return tuple(ind)


@contextlib.contextmanager
def suppress_warnings(msg=None):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', msg)
        yield
