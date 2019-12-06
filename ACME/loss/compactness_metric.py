import torch
from ..geometry.compactness import *


def compactness_metric(P, i=None):
    """
    Measures the compactness metric for the given points set

    Parameters
    ----------
    P : Tensor
        a (...,N,D,) points set tensor

    Returns
    -------
    Tensor
        a (1,) compactness metric tensor
    """

    if i is None:
        return torch.sum(compactness(P))
    return torch.sum(compactness(P[i]))
