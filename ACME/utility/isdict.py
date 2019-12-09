from .istype import *


def isdict(*obj):
    """
    Returns whether or not the input is a dict

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are dicts, False otherwise
    """

    return istype(dict, *obj)
