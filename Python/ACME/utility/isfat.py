from .row import *
from .col import *
from .ismatrix import *

def isfat(*tensors):
    """
    Returns whether or not the input tensor is a fat matrix

    A fat matrix is a matrix where the number of rows is smaller than the columns

    Parameters
    ----------
    *tensors : Tensor...
        a sequence of tensors

    Returns
    -------
    bool
        True if the tensors are fat, False otherwise
    """

    return all([ismatrix(t) and (row(t)<col(t)) for t in tensors])
