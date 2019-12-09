import torch.sparse
from .istype import *


def issparse(*obj):
    """
    Returns whether or not the inputs are PyTorch sparse tensors

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are PyTorch sparse Tensors, False otherwise
    """

    return istype(torch.sparse.FloatTensor, *obj)
