import torch
from .affine2rotm import *


def affine2rotation(M):
    """
    Returns the rotation component of the input affine matrix

    Parameters
    ----------
    M : Tensor
        a (4,4) tensor

    Returns
    -------
    Tensor
        a (3,3) tensor
    """

    u, _, v = torch.svd(affine2rotm(M))
    return torch.mm(u, v)
