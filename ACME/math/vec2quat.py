from .cart2homo import *


def vec2quat(V, dim=-1):
    """
    Converts a vector in quaternion form

    Parameters
    ----------
    V : Tensor
        a (3,) or (N,3,) vector tensor
    dim : int (optional)
        the dimension along the vectors should be converted

    Returns
    -------
    Tensor
        the (4,) or (N,4,) quaternion tensor
    """

    return cart2homo(V, w=0, dim=dim)
