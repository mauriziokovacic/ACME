import torch
from .istype import *


def istorch(*obj):
    """
    Returns whether or not the input is a PyTorch tensor

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are PyTorch Tensors, False otherwise
    """

    return istype(torch.Tensor, *obj)
