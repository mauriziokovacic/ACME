import numpy
from .istype import *


def isnumpy(*obj):
    """
    Returns whether or not the input is a numpy array

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are Numpy arrays, False otherwise
    """

    return istype(numpy.ndarray, *obj)
