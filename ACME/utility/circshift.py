import numpy
import torch
from .isnumpy import *
from .istorch import *

def circshift(tensor,k,dim=None):
    """
    Circularly shifts the input tensor k times along the given dimension

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    k : int
        the number of shifts to perform
    dim : int (optional)
        the dimension along the shift is performed (default is None)

    Returns
    -------
    Tensor
        the tensor with shifted values

    Raises
    ------
    AssertionError
        if input tensor is neither a Numpy or PyTorch tensor
    """

    if isnumpy(tensor):
        return numpy.roll(tensor,k,axis=dim)
    if istorch(tensor):
        return torch.roll(tensor,k,dims=dim)
    assert False, 'Unknown data type'
