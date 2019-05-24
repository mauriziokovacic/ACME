from .istensor import *
from .flatten  import *

def isnan(a):
    """
    Returns whether or not the input is nan

    Parameters
    ----------
    a : int or float or tensor
        an input

    Returns
    -------
    bool
        True if the input is nan, False otherwise
    """

    if istensor(a):
        a = flatten(a)
    return any(a!=a)
