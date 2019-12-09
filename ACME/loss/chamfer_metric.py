from ..math.unrooted_norm import *
from ..math.knn           import *


def chamfer_metric(A, B, alpha=1, beta=1):
    """
    Returns the chamfer metric for the given points sets

    Parameters
    ----------
    A : Tensor
        the first points set tensor
    B : Tensor
        the second points set tensor
    alpha : float
        multiplier of the first term
    beta : float
        multiplier of the second term

    Returns
    -------
    Tensor
        the (1,) metric tensor
    """
    return alpha * torch.mean(knn(A, B, 1, distFcn=sqdistance)[1]) +\
           beta  * torch.mean(knn(B, A, 1, distFcn=sqdistance)[1])


def pixel2mesh_chamfer(A, B):
    """
    Returns the chamfer metric as computed in Pixel2Mesh

    Parameters
    ----------
    P : Tensor
        the points set tensor
    T : LongTensor
        the topology tensor
    alpha : float
        multiplier of the first term
    beta : float
        multiplier of the second term

    Returns
    -------
    Tensor
        the (1,) metric tensor
    """
    return chamfer_metric(A, B, alpha=1, beta=0.55)
