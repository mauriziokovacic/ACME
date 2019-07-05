from .row import *
from .col import *
from .ismatrix import *


def isskinny(*tensors):
    """
    Returns whether or not the input tensor is a skinny matrix

    A fat matrix is a matrix where the number of columns is smaller than the rows

    Parameters
    ----------
    *tensors : Tensor
        a sequence of tensors

    Returns
    -------
    bool
        True if all the tensors are skinny, False otherwise
    """

    return all([ismatrix(t) and (row(t) > col(t)) for t in tensors])
