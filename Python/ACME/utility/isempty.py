from .numel import *

def isempty(A):
    """
    Returns whether or not the input Tensor has no elements

    Parameters
    ----------
    A : Tensor
        a tensor

    Returns
    -------
    bool
        True if the tensor is empty, False otherwise
    """

    if isnone(A):
        return True
    return numel(A)==0
