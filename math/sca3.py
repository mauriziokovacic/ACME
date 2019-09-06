from .eye         import *
from .rotm2affine import *


def sca3(s, affine=False):
    """
    Creates a 3D scaling matrix with the specified scaling factors

    Parameters
    ----------
    s : Tensor
        a three elements tensor
    affine : bool (optional)
        if True returns a (4,4) matrix, (3,3) otherwise
    """

    S = eye(3, dtype=s.dtype, device=s.device)*s.flatten()
    if affine:
        S = rotm2affine(S)
    return S
