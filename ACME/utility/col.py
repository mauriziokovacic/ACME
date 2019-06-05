from .size import *
from .ndim import *

def col(A):
    """
    Returns the number of columns of the given Tensor

    Parameters
    ----------
    A : Tensor
        A tensor/matrix

    Returns
    -------
    int
        the number of columns
    """

    if ndim(A)>=2:
        return size(A)[1]
    return 0
