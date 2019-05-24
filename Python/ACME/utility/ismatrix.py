from .size     import *
from .isscalar import *

def ismatrix(a):
    """
    Returns whether or not the input is a matrix

    Parameters
    ----------
    a : object
        the input object

    Returns
    -------
    bool
        True if input is a tensor in matrix form, False otherwise
    """

    s = size(a)
    if isscalar(s):
        return False
    return len(s)==2
