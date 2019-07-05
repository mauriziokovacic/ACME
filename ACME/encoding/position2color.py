import torch
from ..math.normalize import *


def position2color(P, min=None, max=None):
    """
    Converts the input points set into colors

    Parameters
    ----------
    P : Tensor
        the input points set tensor
    min : float (optional)
        the minimum value for the points set. If None it will be automatically computed (default is None)
    max : float (optional)
        the maximum value for the points set. If None it will be automatically computed (default is None)

    Returns
    -------
    Tensor
        the color tensor
    """

    if min is None:
        min = torch.min(P)[0].item()
    if max is None:
        max = torch.max(P)[0].item()
    return normalize(P, min=min, max=max)
