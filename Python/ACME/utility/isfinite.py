from .isnan import *
from .isinf import *

def isfinite(a):
    """
    Returns whether or not the input is finite

    Parameters
    ----------
    a : int or float or tensor
        an input

    Returns
    -------
    bool
        True if the input is finite, False otherwise
    """

    return (not isnan(a)) and (not isinf(a))
