import numpy
import torch
from .isnumpy import *
from .istorch import *


def reshape(tensor, shape, **kwargs):
    """
    Reshapes the input tensor into the given shape

    Parameters
    ----------
    tensor : Tensor
        an input Numpy or PyTorch tensor
    shape : list or tuple
        the new shape of the input tensor
    **kwargs : ...
        any argument taken from the Numpy version of reshape

    Returns
    -------
    Tensor
        the reshaped tensor
    """

    if isnumpy(tensor):
        return numpy.reshape(tensor, shape, **kwargs)
    if istorch(tensor):
        return torch.reshape(tensor, shape)
    assert False, 'Unknown data type'
