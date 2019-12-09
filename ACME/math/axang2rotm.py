from ..utility.FloatTensor import *
from .cos                  import *
from .sin                  import *


def axang2rotm(U, theta):
    """
    Creates a rotatiom matrix from a given normalized axis and angle in radians

    Parameters
    ----------
    U : Tensor
        normalized 3D rotation axis

    theta : float
        rotation angle in radians

    Returns
    -------
    Tensor
        the (3,3) rotation matrix
    """

    x, y, z = U
    c = cos(theta)
    s = sin(theta)
    t = 1-c
    R = FloatTensor([[ c+x**2*t, x*y*t-z*s, x*z*t+y*s],
                     [y*x*t+z*s,  c+y**2*t, y*z*t-x*s],
                     [z*x*t-y*s, z*y*t+x*s,  c+z**2*t]],device=U.device)
    return R
