from .row import *
from .col import *
from .ismatrix import *


def issquare(*tensors):
    """
    Returns whether or not the input tensor is a square matrix

    A fat matrix is a matrix where the number of rows is equal to the columns

    Parameters
    ----------
    *tensors : Tensor...
        a sequence of tensors

    Returns
    -------
    bool
        True if the tensors are square, False otherwise
    """

    return all([ismatrix(t) and (row(t) == col(t)) for t in tensors])
