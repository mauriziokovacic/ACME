import torch
from ..utility.col import *


def affine2translation(M):
    """
    Returns the translation component of the input affine matrix

    Parameters
    ----------
    M : Tensor
        a (4,4) tensor

    Returns
    -------
    Tensor
        a (3,) tensor representing the translation
    """

    if col(M) < 4:
      return torch.zeros(3, dtype=M.dtype, device=M.device)
    return M[0:3, 3]
