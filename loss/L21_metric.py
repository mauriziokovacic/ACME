from ..math.unrooted_norm import *


def L21_metric(N, A, n):
    """
    Computes the L^2,1 metric for the given data

    Parameters
    ----------
    N : Tensor
        the (N,D,) polygon normal tensor
    A : Tensor
        the (N,) polygon area tensor
    n : Tensor
        the (N,D,) proxy normal tensor

    Returns
    -------
    Tensor
        the (N,) metric tensor
    """

    return sqdistance(N, n) * A
