from .isnone   import *
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

    if isnone(A) or isscalar(A):
        return 0
    if islist(A) or istuple(A):
        return len(A)
    return A.shape
