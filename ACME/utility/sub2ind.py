import torch
from .prod import *

def sub2ind(size, I):
    """
    Converts subscripts to linear indices

    Parameters
    ----------
    size : iterable
        the size of the target tensor
    I : LongTensor
        the (N,M,) subscripts tensor

    Returns
    -------
    LongTensor
        the linear indices of the given subscripts

    Raises
    ------
    AssertionError
        if rows in I are less than dimensions in size
    """

    assert len(size) == I.size(0), 'Tensors should match dimensions'
    return torch.arange(0, prod(size), dtype=torch.long, device=I.device).view(size)[tuple(I)]

