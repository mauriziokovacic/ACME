import torch


def flipud(tensor):
    """
    Flips a given tensor along the first dimension (up to down)

    Parameters
    ----------
    tensor
        a tensor at least two-dimensional

    Returns
    -------
    Tensor
        the flipped tensor
    """

    return torch.flip(tensor, dims=[0])
