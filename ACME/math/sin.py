import math
import torch
from utility import *

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

    if istorch(theta):
        return torch.sin(theta)
    return math.sin(theta)
