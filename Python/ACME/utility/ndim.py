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

    if A is None:
        return 0
    return len(size(A))
