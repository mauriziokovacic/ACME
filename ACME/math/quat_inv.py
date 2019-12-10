from .quat_conj import *
from .quat_norm import *


def quat_inv(Q):
    """
    Returns the inverse quaternion

    Parameters
    ----------
    Q : Tensor
        the (4,) or (N,4,) quaternion tensor

    Returns
    -------
    Tensor
        the (4,) or (N,4,) inverse quaternion tensor
    """

    return quat_conj(Q) / quat_norm(Q)
