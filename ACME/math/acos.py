import math
import torch
from utility import *

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

    if istorch(theta):
        return torch.acos(theta)
    return math.acos(theta)
