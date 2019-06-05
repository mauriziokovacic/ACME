import numpy
import torch
from .isnumpy import *
from .istorch import *

def transpose(tensor):
    """
    Transposes the input tensor

    Parameters
    ----------
    tensor : Tensor
        a nxm tensor

    Returns
    -------
    Tensor
        a mxn tensor

    Raises
    ------
    AssertError
        if tensor does not belongs to Numpy or PyTorch
    """

    if isnumpy(tensor):
        return numpy.transpose(tensor)
    if istorch(tensor):
        return torch.t(tensor)
    assert False, 'Unknown data type'
