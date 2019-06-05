import torch

def fliplr(tensor):
    """
    Flips a given tensor along the second dimension (left to right)

    Parameters
    ----------
    tensor
        a tensor at least two-dimensional

    Returns
    -------
    Tensor
        the flipped tensor
    """

    return torch.flip(tensor, dims=[1])
