import torch
from ..math.knn           import *
from ..math.unrooted_norm import *


def hausdorff_metric(A, B):
    """
    Returns the Hausdorff metric for the given two points sets

    Parameters
    ----------
    A : Tensor
        the first points set tensor
    B : Tensor
        the second points set tensor

    Returns
    -------
    Tensor
        the (1,) metric tensor
    """

    return torch.sum(knn(A, B, 1, distFcn=sqdistance)[1]) +\
           torch.sum(knn(B, A, 1, distFcn=sqdistance)[1])
