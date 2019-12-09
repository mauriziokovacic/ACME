from .istype import *


def istuple(*obj):
    """
    Returns whether or not the input is a tuple

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are tuple, False otherwise
    """

    return istype(tuple, *obj)
