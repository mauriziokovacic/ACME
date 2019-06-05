import torch

def color2position(C,min=None,max=None):
    """
    Converts the input points set into colors

    Parameters
    ----------
    C : Tensor
        the input color tensor
    min : float (optional)
        the minimum value for the points set. If None it will be set to -1 (default is None)
    max : float (optional)
        the maximum value for the points set. If None it will be set to +1 (default is None)

    Returns
    -------
    Tensor
        the points set tensor
    """

    if min is None:
        min = -1
    if max is None:
        max = 1
    return torch.add(torch.mul(C,max-min),min)
