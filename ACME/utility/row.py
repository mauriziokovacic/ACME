from .size import *
from .ndim import *


def row(A):
    """
    Returns the number of rows in the given Tensor

    Parameters
    ----------
    A : Tensor
        A tensor/matrix

    Returns
    -------
    int
        the number of rows
    """

    if ndim(A) >= 1:
        return size(A)[0]
    return 0
