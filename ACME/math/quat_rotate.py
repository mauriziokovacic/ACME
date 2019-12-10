from .quat_inv      import *
from .quat_multiply import *
from .vec2quat      import *


def quat_rotate(Q, P):
    """
    Rotates a point by a given quaternion

    Parameters
    ----------
    Q : Tensor
        a (4,) or (N,4,) quaternion tensor
    P : Tensor
        a (3,) or (N,3,) point set tensor

    Returns
    -------
    Tensor
        a (3,) or (N,3,) rotated point set tensor
    """

    return quat_multiply(quat_multiply(Q, vec2quat(P)), quat_inv(Q))
