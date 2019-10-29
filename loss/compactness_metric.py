import torch
from ..geometry.compactness import *


def compactness_metric(P):
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

    return torch.sum(compactness(P))
