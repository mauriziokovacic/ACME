import torch
from .triangle_angle import *


def triangle_ratio(P, T):
    """
    Returns the ratio between the minimum and maximum angles of the given triangles

    Parameters
    ----------
    P : Tensor
        the input points set tensor
    T : LongTensor
        the topology tensor

    Returns
    -------
    Tensor
        the triangles ratio
    """

    A = torch.cat(triangle_angle(P, T), dim=1)
    return torch.min(A, 1, keepdim=True)[0]/torch.max(A, 1, keepdim=True)[0]
