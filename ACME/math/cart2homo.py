import torch
from ..utility.row            import *
from ..utility.ConstantTensor import *


def cart2homo(P, w=1, dim=-1):
    """
    Converts a points set from cartersian to homogeneous coordinates

    Parameters
    ----------
    P : Tensor
        the cartesian coordinates
    w : float (optional)
        the fourth component
    dim : int (optional)
        the dimension to add the w component (default is -1)

    Returns
    -------
    Tensor
        the homogeneous coordinates
    """

    s = list(P.shape)
    s[dim] = 1
    return torch.cat((P, ConstantTensor(w, *s, dtype=P.dtype, device=P.device)), dim=dim)
