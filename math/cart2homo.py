import torch
from ..utility.row            import *
from ..utility.ConstantTensor import *


def cart2homo(P, w=1):
    """
    Converts a points set from cartersian to homogeneous coordinates

    Parameters
    ----------
    P : Tensor
        the cartesian coordinates
    w : float (optional)
        the fourth component

    Returns
    -------
    Tensor
        the homogeneous coorodinates
    """

    return torch.cat((P, ConstantTensor(w, row(P), 1, dtype=P.dtype, device=P.device)), dim=1)
