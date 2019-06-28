import torch
from .norm              import *
from .prepare_broadcast import *

def knn(A, B, k, distFcn=distance):
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
    distFcn : callable (optional)
        the distance function to use (default is euclidean norm)
    """

    a, b = prepare_broadcast(A, B)
    d, i = torch.topk(-distFcn(a, b, dim=-1), k, dim=1)
    return i, -d
