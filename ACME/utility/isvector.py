from .assertion import *
from .numel     import *
from .istensor  import *


def isvector(*tensors):
    """
    Returns whether or not the input tensor is a vector

    A vector is a tensor where the number of elements is equal to one of the dimensions

    Parameters
    ----------
    *tensor : Tensor...
        a sequence of tensors

    Returns
    -------
    bool
        True if the tensors are vectors, False otherwise

    Raises
    ------
    AssertError
        if inputs are not tensors
    """

    [assertion(istensor(t), 'Inputs must be tensors') for t in tensors]
    return all([any([numel(t) == d for d in t.shape]) for t in tensors])
