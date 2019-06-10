import math
import numpy
import torch
from ACME.utility.isnumpy import *
from ACME.utility.istorch import *

def sin(theta):
    """
    Computes the sine of the input thetas

    Parameters
    ----------
    theta : int, float or Tensor
        the value of the angle in radians

    Returns
    -------
    float or Tensor
        the sine of the input
    """

    if isnumpy(theta):
        return numpy.sin(theta)
    if istorch(theta):
        return torch.sin(theta)
    return math.sin(theta)
