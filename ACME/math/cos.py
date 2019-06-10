import math
import numpy
import torch
from ACME.utility.isnumpy import *
from ACME.utility.istorch import *

def cos(theta):
    """
    Computes the cosine of the input thetas

    Parameters
    ----------
    theta : int, float or Tensor
        the value of the angle in radians

    Returns
    -------
    float or Tensor
        the cosine of the input
    """

    if isnumpy(theta):
        return numpy.cos(theta)
    if istorch(theta):
        return torch.cos(theta)
    return math.cos(theta)
