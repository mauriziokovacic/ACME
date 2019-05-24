from .isnumpy import *
from .istorch import *

def istensor(A):
    """
    Returns whether or not the input is a Numpy or PyTorch tensor

    Parameters
    ----------
    obj : object
        any object

    Returns
    -------
    bool
        True if the input is a Numpy or PyTorch Tensor, False otherwise
    """

    return isnumpy(A) or istorch(A)
