from .quat_scalar import *
from .quat_vector import *
from .dot         import *
from .cross       import *


def quat_multiply(Q1, Q2, dim=-1):
    """
    Multiplies two quaternions together

    Parameters
    ----------
    Q1 : Tensor
        the (4,) or (N,4,) quaternion tensor
    Q2 : Tensor
        the (4,) or (N,4,) quaternion tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the (4,) or (N,4,) quaternion tensor
    """

    v1 = quat_vector(Q1, dim=dim)
    s1 = quat_scalar(Q1, dim=dim)
    v2 = quat_vector(Q2, dim=dim)
    s2 = quat_scalar(Q2, dim=dim)
    return torch.cat((s1 * s2 - dot(v1, v2, dim=dim), s1 * v2 + s2 * v1 + cross(v1, v2, dim=dim)), dim=dim)
