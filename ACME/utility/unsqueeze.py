import numpy
import torch
from .isnumpy import *
from .istorch import *

def unsqueeze(tensor,dim):
    """
    Add a dimension of value 1 in the input tensor at the specified position

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    dim : int
        dimension to expand

    Returns
    -------
    Tensor
        the unsqueezed input

    Raises
    ------
    AssertError
        if tensor does not belongs to Numpy or PyTorch
    """

    if isnumpy(tensor):
        return numpy.expand_dims(tensor,axis=dim)
    if istorch(tensor):
        return tensor.unsqueeze(dim)
    assert False, 'Unknown data type'
