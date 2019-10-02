from ..utility.matmul import *


def cage_deformation(C, W):
    """
    Returns the points set defined by the cage points

    Parameters
    ----------
    C : Tensor
        the (M,3,) cage points tensor
    W : Tensor or SparseTensor
        the (N,M,) weights tensor

    Returns
    -------
    Tensor
        a (N,3,) points set tensor
    """

    return matmul(W, C)
