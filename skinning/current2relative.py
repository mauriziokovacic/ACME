import torch


def current2relative(rest_pose, current_pose):
    """
    Returns the transforms to pass from a rest pose to a current one

    Parameters
    ----------
    rest_pose : Tensor
        a (N,K,K,) tensor representing the rest pose
    current_pose : Tensor
        a (N,K,K,) tensor representing the current pose

    Returns
    -------
    Tensor
        a (N,K,K,) tensor representing the relative transformations
    """
    
    return torch.matmul(current_pose, torch.inverse(rest_pose))
