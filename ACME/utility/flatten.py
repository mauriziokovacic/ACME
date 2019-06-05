import numpy
import torch
from .isnumpy import *
from .istorch import *

def flatten(tensor):
    """
    Flattens a given tensor

    Parameters
    ----------
    tensor : Tensor
        an input tensor

    Returns
    -------
    Tensor
        the flatten version of the input

    Raises
    ------
    AssertError
        if tensor is not a Numpy or PyTorch tensor
    """

    if isnumpy(tensor):
        return numpy.ravel(tensor)
    if istorch(tensor):
        return torch.flatten(tensor)
    assert False, 'Unknown data type'
