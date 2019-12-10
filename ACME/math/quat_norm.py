from .norm import *


def quat_norm(Q, dim=-1):
    """
    Returns the quaternion norm

    Parameters
    ----------
    Q : Tensor
        the (4,) or (N,4,) quaternion tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the (1,) or (N,) norm tensor
    """
    return norm(Q, dim=dim)
