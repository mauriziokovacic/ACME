import numpy

def isnumpy(A):
    """
    Returns whether or not the input is a numpy array

    Parameters
    ----------
    obj : object
        any object

    Returns
    -------
    bool
        True if the input is a numpy array, False otherwise
    """

    return isinstance(obj,numpy.ndarray)
