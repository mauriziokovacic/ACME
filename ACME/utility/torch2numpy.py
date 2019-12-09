import numpy
import torch
from .isnumpy import *
from .istorch import *


def torch2numpy(tensor):
    """
    Converts the input tensor from PyTorch to Numpy.

    Parameters
    ----------
    tensor : Tensor
        the input tensor

    Returns
    -------
    Tensor
        the converted tensor

    Raises
    ------
    AssertionError
        if input tensor is neither a Numpy or PyTorch tensor
    """

    if isnumpy(tensor):
        return tensor
    if istorch(tensor):
        return tensor.cpu().detach().numpy()
    assert False, 'Unknown data type'
