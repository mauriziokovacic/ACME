import math
import numpy
import torch
from ..utility.isnumpy import *
from ..utility.istorch import *

def floor(a):
    """
    Returns the floor of the input

    Parameters
    ----------
    a : scalar or Numpy or PyTorch tensor
        the input value

    Returns
    -------
    scalar or Numpy or PyTorch tensor
        the floored input
    """

    if isnumpy(a):
        return numpy.floor(a)
    if istorch(a):
        return torch.floor(a)
    return math.floor(a)
