import torch
from ..utility.istorch import *

def linint(A,B,t):
    """
    Computes the linear interpolation between the two input at the specified parameter

    Parameters
    ----------
    A : Tensor
        the first input tensor
    B : Tensor
        the second input tensor
    t : float
        the interpolation parameter

    Returns
    -------
    Tensor
        the inpterpolated value
    """

    if istorch(A,B):
        return torch.lerp(A,B,t)
    return A*(1-t)+B*t
