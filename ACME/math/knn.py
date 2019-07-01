import torch
from .prepare_broadcast import *

def knn(A, B, k, distFcn):
    """
    Returns the indices of the k-nearest neighbors of A in B

    Parameters
    ----------
    A : Tensor
        a (N,F,) tensor
    B : Tensor
        a (M,F,) tensor
    k : int
        the number of neighbors to find
    distFcn : callable
        the distance function to use

    Returns
    -------
    (LongTensor, Tensor)
        the indices of the k neaighbors and their distance

    Raises
    ------
    AssertionError
        if k is lower than 1
    """

    assert k>0, 'k value must be greater than 0'
    a, b = prepare_broadcast(A, B)
    if k>1:
        d, i = torch.topk(-distFcn(a, b, dim=-1), k, dim=1)
        return i, -d
    d, i = torch.min(distFcn(a, b, dim=-1), 1, keepdim=True)
    return i, d
