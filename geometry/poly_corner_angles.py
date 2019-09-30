from ..math.acos          import *
from .poly_corner_cosines import *


def poly_corner_angles(P, T):
    """
    Returns the per polygon corner angles

    Parameters
    ----------
    P : Tensor
        a (N, D,) points set tensor
    T : LongTensor
        a (M, T,) topology tensor

    Returns
    -------
    Tensor
        the (T, M,) corner angles tensor
    """

    return acos(poly_corner_cosines(P, T))
