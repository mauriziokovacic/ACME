from ..utility.isnumpy   import *
from ..utility.istorch   import *
from ..utility.row       import *
from ..utility.squeeze   import *
from ..utility.unsqueeze import *
from .cos                import *
from .sin                import *


def eul2rotm(theta):
    """
    Returns a rotation matrix from the given euler angles

    Parameters
    ----------
    theta : Tensor
        a (3,) or (N,3,) euler angles tensor

    Returns
    -------
    Tensor
        the (3,3,) or (N,3,3,) rotation matrices tensor
    """

    t = theta
    if ndim(theta) == 1:
        t = unsqueeze(theta, 0)
    c = cos(t)
    s = sin(t)
    if isnumpy(theta):
        R = numpy.zeros((row(t), 3, 3), dtype=float)
    if istorch(theta):
        R = torch.zeros((row(t), 3, 3), dtype=torch.float, device=theta.device)
    R[:, 0, 0] =  c[:, 1] * c[:, 2]
    R[:, 0, 1] = -c[:, 1] * s[:, 2]
    R[:, 0, 2] =  s[:, 1]
    R[:, 1, 0] =  c[:, 0] * s[:, 2] + c[:, 2] * s[:, 0] * s[:, 1]
    R[:, 1, 1] =  c[:, 0] * c[:, 2] - s[:, 0] * s[:, 1] * s[:, 2]
    R[:, 1, 2] = -c[:, 1] * s[:, 0]
    R[:, 2, 0] =  s[:, 0] * s[:, 2] - c[:, 0] * c[:, 2] * s[:, 1]
    R[:, 2, 1] =  c[:, 2] * s[:, 0] + c[:, 0] * s[:, 1] * s[:, 2]
    R[:, 2, 2] =  c[:, 0] * c[:, 1]
    return squeeze(R)
