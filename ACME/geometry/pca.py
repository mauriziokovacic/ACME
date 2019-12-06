import torch
from .barycenter import *


def pca(P, dim=0):
    """
    Returns the principal component analysis of the given input set

    Parameters
    ----------
    P : Tensor
        the input point set tensor
    dim : int (optional)
        the dimension along the pca is computed

    Returns
    -------
    (Tensor,Tensor)
        the barycenter of the point set and its principal directions in matrix form
    """

    B       = barycenter(P, dim=dim)
    p       = P-B
    C       = torch.mm(torch.t(p), p)
    U, S, V = torch.svd(C)
    return B, torch.t(V)
