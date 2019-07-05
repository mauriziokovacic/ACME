import torch
from ..utility.unique     import *
from ..utility.accumarray import *
from .poly2edge           import *


def boundary(T):
    """
    Returns the boundary edges of the given input topology

    Parameters
    ----------
    T : LongTensor
        the input topology tensor

    Returns
    -------
    LongTensor
        the boundary edge tensor
    """

    E       = poly2edge(T)[0]
    _, j, e = unique(torch.sort(E, 1)[0], ByRows=True)
    E       = E[:, j[accumarray(e, 1) == 1]]
    return E
