import math
import torch
from utility import *

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

    if istorch(theta):
        return torch.cos(theta)
    return math.cos(theta)
