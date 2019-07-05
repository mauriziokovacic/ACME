import torch


def normal2color(N):
    """
    Converts a given normal into a color

    Parameters
    ----------
    N : Tensor
        the input (n,3) normal tensor

    Returns
    -------
    Tensor
        the color coded normal
    """

    return torch.add(torch.mul(N, 0.5), 0.5)
