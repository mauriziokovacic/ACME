import torch
from .angle import *
from .cross import *


def signed_angle(A, B, N, dim=1):
    """
    Computes the signed angle between the A and B w.r.t. N, along the specified dimension

    Parameters
    ----------
    A : Tensor
        the first input tensor
    B : Tensor
        the second input tensor
    N : Tensor
        direction along the sign is computed
    dim : int (optional)
        dimension along the angle computation is performed (default is 1)

    Returns
    -------
    Tensor
        the tensor containing the signed angles
    """

    return angle(A, B, dim=dim) * torch.sign(dot(N, cross(A, B, dim=dim)))
