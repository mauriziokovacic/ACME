import numpy


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

    return all([isinstance(o, numpy.ndarray) for o in obj])
