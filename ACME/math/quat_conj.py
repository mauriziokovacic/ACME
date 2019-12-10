import torch


def quat_conj(Q):
    """
    Returns the conjugate of the given quaternion

    Parameters
    ----------
    Q : Tensor
        the (4,) or (N,4,) quaternion tensor

    Returns
    -------
    Tensor
        the (4,) or (N,4,) conjugate quaternion tensor
    """

    return Q * torch.tensor([-1, -1, -1, 1])
