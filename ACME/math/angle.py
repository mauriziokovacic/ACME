import torch
from utility.clamp import *
from .acos import *

def angle(A,B,dim=1):
    """
    Computes the angle in radians between the inputs along the specified dimension

    Parameters
    ----------
    A : Tensor
        first input tensor
    B : Tensor
        second input tensor
    dim : int (optional)
        dimension along the angle is computed (default is 1)

    Returns
    -------
    Tensor
        the tensor containing the angle between the inputs
    """

    return acos(clamp(dot(A,B,dim=dim),-1,1))
