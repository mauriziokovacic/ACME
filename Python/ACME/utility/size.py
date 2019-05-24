from .isscalar import *
from .islist   import *
from .istuple  import *

def size(A):
    """
    Returns the size of each dimension of the input Tensor

    Parameters
    ----------
    A : Tensor
        A tensor/matrix

    Returns
    -------
    list
        the size of each dimension of the input tensor
    """

    if A is None:
        return 0
    if isscalar(A):
        return 1
    if islist(A) or istuple(A):
        return len(A)
    return A.shape
