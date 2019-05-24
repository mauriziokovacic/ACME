from .isscalar import *
from .size import *

def ndim(A):
    """
    Returns the number of dimensions of the input Tensor

    Parameters
    ----------
    A : Tensor
        a tensor/matrix

    Returns
    -------
    int
        The number of dimensions of the input tensor
    """

    s = size(A)
    if isscalar(s):
        return s
    return len(s)
