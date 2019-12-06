import numpy
import torch
from .isnumpy import *
from .istorch import *


def squeeze(tensor):
    """
    Removes all the dimensions with value 1 of the input tensor

    Parameters
    ----------
    tensor : Tensor
        the input tensor

    Returns
    -------
    Tensor
        the squeezed input

    Raises
    ------
    AssertError
        if tensor does not belongs to Numpy or PyTorch
    """

    if isnumpy(tensor):
        return numpy.squeeze(tensor)
    if istorch(tensor):
        return tensor.squeeze()
    assert False, 'Unknown data type'
