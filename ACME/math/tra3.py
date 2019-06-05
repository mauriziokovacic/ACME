from .eye import *


def tra3(t):
    """
    Creates a 3D translation matrix from a given vector

    Parameters
    ----------
    t : Tensor
        a three elements tensor

    Returns
    -------
    Tensor
        a (4,4) translation matrix
    """

    T = eye(4,dtype=t.dtype,device=t.device)
    T[0:3,3] = t.flatten()
    return T
