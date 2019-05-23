from . import size
from functools import reduce

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

    return reduce((lambda a, b : a*b), size(A))
