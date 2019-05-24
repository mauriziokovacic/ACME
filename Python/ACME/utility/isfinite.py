from .isnan import *
from .isinf import *

def isfinite(*obj):
    """
    Returns whether or not the input is finite

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are finite, False otherwise
    """

    return (not isnan(*obj)) and (not isinf(*obj))
