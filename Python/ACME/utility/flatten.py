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
    """

    if isnumpy(tensor):
        return numpy.ravel(tensor)
    return torch.flatten(tensor)
