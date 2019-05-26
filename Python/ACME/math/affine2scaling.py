import torch
from .affine2rotm     import *
from .affine2rotation import *

def affine2scaling(M):
    """
    Returns the scaling component of the input affine matrix

    Parameters
    ----------
    M : Tensor
        a (4,4) tensor

    Returns
    -------
    Tensor
        a (3,3) tensor representing the scaling
    """

    return torch.diag(torch.mm(affine2rotm(M),torch.t(affine2rotation(M))))
