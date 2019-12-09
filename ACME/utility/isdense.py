import torch
from .istype import *


def isdense(*obj):
    """
    Returns whether or not the inputs are PyTorch dense tensors

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are PyTorch dense Tensors, False otherwise
    """

    return istype(torch.FloatTensor, *obj)
