from .isscalar import *
from .size     import *
from .prod     import *


def numel(A):
    """
    Returns the number of elements contained in the given Tensor

    Parameters
    ----------
    A : Tensor
        a tensor/matrix
    Returns
    -------
    int
        the number of elements in the given tensor
    """

    s = size(A)
    if isscalar(s):
        return 1
    return prod(s)
