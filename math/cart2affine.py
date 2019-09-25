import torch
from ..utility.row            import *
from ..utility.ConstantTensor import *


def cart2affine(P, w=1):
    """
    Converts a points set from cartersian to affine coordinates

    Parameters
    ----------
    P : Tensor
        the affine coordinates
    w : float (optional)
        the fourth component

    Returns
    -------
    Tensor
        the affine coorodinates
    """

    return torch.cat((P, ConstantTensor(w, row(P), 1, dtype=P.dtype, device=P.device)), dim=1)