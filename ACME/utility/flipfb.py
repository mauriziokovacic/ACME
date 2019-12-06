import torch


def flipfb(tensor):
    """
    Flips a given tensor along the third dimension (front to back)

    Parameters
    ----------
    tensor
        a tensor at least three-dimensional

    Returns
    -------
    Tensor
        the flipped tensor
    """

    return torch.flip(tensor, dims=[2])
