from ..math.knn           import *
from ..math.unrooted_norm import *


def hausdorff_metric(A, B):
    """
    Returns the Hausdorff metric for two given tensors

    Parameters
    ----------
    A : Tensor
        a (M,F,) tensor
    B : Tensor
        a (N,F,) tensor

    Returns
    -------
    Tensor
        a (1,) metric tensor
    """

    return torch.sum(knn(A, B, 1, distFcn=sqdistance)[1].squeeze()) + \
           torch.sum(knn(B, A, 1, distFcn=sqdistance)[1].squeeze())
