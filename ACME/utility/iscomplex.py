from .istype import *


def iscomplex(*obj):
    """
    Returns whether or not the input is a complex

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are complex, False otherwise
    """

    return istype(complex, *obj)
