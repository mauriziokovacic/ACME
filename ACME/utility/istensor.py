from .isnumpy import *
from .istorch import *


def istensor(*obj):
    """
    Returns whether or not the inputs are Numpy or PyTorch tensors

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are Numpy or PyTorch Tensors, False otherwise
    """

    return isnumpy(*obj) or istorch(*obj)
