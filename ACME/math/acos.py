import math
import numpy
import torch
from ..utility.isnumpy import *
from ..utility.istorch import *


def acos(theta):
    """
    Computes the arccosine of the input thetas

    Parameters
    ----------
    theta : int, float or Tensor
        the value of the angle cosine

    Returns
    -------
    float or Tensor
        the angle of the input
    """

    if isnumpy(theta):
        return numpy.acos(theta)
    if istorch(theta):
        return torch.acos(theta)
    return math.acos(theta)
