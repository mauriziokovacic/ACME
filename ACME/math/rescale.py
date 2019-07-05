import torch
from .normalize import *


def rescale(tensor, min=None, max=None):
    """
    Rescales the tensor values in range [min-max]

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    min : int or float (optional)
        the new minimum value. If None, min(tensor) will be used instead (default is None)
    max : int or float (optional)
        the new maximum value. If None, max(tensor) will be used instead (default is None)

    Returns
    -------
    Tensor
        the rescaled version of the input tensor
    """

    if min is None:
        min = 0
    if max is None:
        max = 1
    return torch.add(torch.mul(normalize(tensor), max-min), min)
