from .numel import *


def isempty(*obj):
    """
    Returns whether or not the input Tensor has no elements

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are empty, False otherwise
    """

    return all([numel(o) == 0 for o in obj])
