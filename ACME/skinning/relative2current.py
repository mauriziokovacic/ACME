import torch


def relative2current(rest_pose, relative_pose):
    """
    Returns the current pose from a rest pose to a relative one

    Parameters
    ----------
    rest_pose : Tensor
        a (N,K,K,) tensor representing the rest pose
    relative_pose : Tensor
        a (N,K,K,) tensor representing the relative pose

    Returns
    -------
    Tensor
        a (N,K,K,) tensor representing the current pose
    """

    return torch.matmul(relative_pose, rest_pose)
