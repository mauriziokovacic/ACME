import torch
from ..utility.ConstantTensor import *


def homo2cart(P, dim=-1):
    """
    Converts a points set from homogeneous coordinates to standard cartesian

    Parameters
    ----------
    P : Tensor
        the homogeneous coordinates
    dim : int (optional)
        the dimension corresponding to the homogeneous component

    Returns
    -------
    Tensor
        the cartesian coordinates
    """

    w = P.shape[dim]-1
    return torch.index_select(P, dim, torch.arange(w, dtype=torch.long, device=P.device)) / \
        torch.index_select(P, dim, torch.tensor([w], dtype=torch.long, device=P.device))
