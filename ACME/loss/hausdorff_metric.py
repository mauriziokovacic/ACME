import torch
from ..math.knn import *

def hausdorff_metric(A, B, distFcn):
    """
    Returns the Hausdorff metric for two given tensors

    Parameters
    ----------
    A : Tensor
        a (M,F,) tensor
    B : Tesnor
        a (N,F,) tensor
    distFcn : callable
        the distance function to be used

    Returns
    -------
    Tensor
        a (1,) metric tensor

    """

    return torch.sum(knn(A, B, 1, distFcn=distFcn)[1])+\
           torch.sum(knn(B, A, 1, distFcn=distFcn)[1])
