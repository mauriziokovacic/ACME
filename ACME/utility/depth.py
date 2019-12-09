from .size import *
from .ndim import *


def depth(A):
    """
    Returns the number of slices/channels of the given Tensor

    Parameters
    ----------
    A : Tensor
        A tensor/matrix

    Returns
    -------
    int
        the number of slices/channels
    """

    if ndim(A) >= 3:
        return size(A)[2]
    return 0
