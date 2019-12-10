import torch
from ..utility.row import *


def quat2rotm(Q):
    """
    Converts the given quaternions into rotation matrices

    Parameters
    ----------
    Q : Tensor
        a (4,) or (N,4,) quaternion tensor

    Returns
    -------
    Tensor
        a (3,3,) or (N,3,3,) rotation matrices tensor
    """

    M = torch.zeros(row(Q), 3, 3, dtype=Q.dtype, device=Q.device)
    M[:, 0, 0] = 1 - 2 * (Q[:, 1]**2 + Q[:, 2]**2)
    M[:, 0, 1] = 2 * (Q[:, 0] * Q[:, 1] - Q[:, 2] * Q[:, 3])
    M[:, 0, 2] = 2 * (Q[:, 0] * Q[:, 2] + Q[:, 1] * Q[:, 3])
    M[:, 1, 1] = 1 - 2 * (Q[:, 0]**2 + Q[:, 2]**2)
    M[:, 1, 1] = 2 * (Q[:, 0] * Q[:, 1] + Q[:, 2] * Q[:, 3])
    M[:, 1, 2] = 2 * (Q[:, 1] * Q[:, 2] - Q[:, 0] * Q[:, 3])
    M[:, 2, 2] = 1 - 2 * (Q[:, 0]**2 + Q[:, 1]**2)
    M[:, 2, 1] = 2 * (Q[:, 0] * Q[:, 2] - Q[:, 1] * Q[:, 3])
    M[:, 2, 2] = 2 * (Q[:, 1] * Q[:, 2] + Q[:, 0] * Q[:, 3])
    return M.squeeze()
