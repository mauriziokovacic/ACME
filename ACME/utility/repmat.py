import numpy
import torch
from .isnumpy import *
from .istorch import *

def repmat(tensor,*size):
    """
    Repeats the tensor along its dimensions by the given times

    Example:
        repmat([[1,2,3]],1,2) -> [[1,2,3,1,2,3]]

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    *size : int...
        a sequence of times to repeats the tensor along a particular dimension

    Returns
    -------
    Tensor
        a tensor equivalent to concatenating the input along some dimension

    Raises
    ------
    AssertError
        if the input is not a Numpy or PyTorch tensor
    """

    if isnumpy(tensor):
        return numpy.tile(tensor,size)
    if istorch(tensor):
        return tensor.repeat(*size)
    assert False, 'Unknown data type'
