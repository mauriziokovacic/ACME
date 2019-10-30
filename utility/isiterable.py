from .hasmethod import *


def isiterable(*obj):
    """
    Returns whether or not the input is iterable

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are iterable, False otherwise
    """

    return all([hasmethod(o, '__iter__') for o in obj])
