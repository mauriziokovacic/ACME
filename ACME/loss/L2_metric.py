from ..geometry.point_plane_distance import *


def L2_metric(P, T, A, X, N):
    """
    Computes the L^2 metric for the given data

    Parameters
    ----------
    P : Tensor
        the (N,D,) points set tensor
    T : LongTensor
        the (3,M,) topology tensor
    A : Tensor
        the (M,) polygon area tensor
    X : Tensor
        the (X,D,) proxy points set tensor
    N : Tensor
        the (X,D,) proxy normal tensor

    Returns
    -------
    Tensor
        the (M,) metric tensor
    """

    d = point_plane_distance(X.unsqueeze(0), N.unsqueeze(0), P[T]).squeeze()
    return (1/6) *\
           (d[:, 0]**2 +
            d[:, 1]**2 +
            d[:, 2]**2 +
            d[:, 0] * d[:, 1] +
            d[:, 0] * d[:, 2] +
            d[:, 1] * d[:, 2]) *\
           A
