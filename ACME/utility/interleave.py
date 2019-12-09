import torch
from .row import *


def interleave(*tensors):
    """
    Interleaves the rows of the given tensors.

    Parameters
    ----------
    *tensors : Tensors...
        a sequence of same size Tensors

    Returns
    -------
    Tensor
        A single two dimensional tensor
    """

    n = len(tensors)
    r = row(tensors[0])
    return torch.reshape(torch.cat(tensors, dim=1), (n*r, *tuple(tensors[0].shape[1:])))
