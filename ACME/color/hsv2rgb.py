from .channel import *


def hsv2rgb(color, dim=-1):
    """
    Converts the given HSV colors in RGB format

    Parameters
    ----------
    color : Tensor
        the HSV color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the RGB color tensor
    """

    H = channel(color, 0, dim=dim)
    S = channel(color, 1, dim=dim)
    V = channel(color, 2, dim=dim)
    C = V * S
    z = torch.zeros_like(C)
    X = C * (1 - torch.abs(torch.fmod(H * 6, 2) - 1))
    m = V - C
    RGB = torch.zeros_like(color)
    RGB = torch.where(H >=   0, torch.cat((C, X, z), dim=dim), RGB)
    RGB = torch.where(H >= 1/6, torch.cat((X, C, z), dim=dim), RGB)
    RGB = torch.where(H >= 1/3, torch.cat((z, C, X), dim=dim), RGB)
    RGB = torch.where(H >= 1/2, torch.cat((z, X, C), dim=dim), RGB)
    RGB = torch.where(H >= 2/3, torch.cat((X, z, C), dim=dim), RGB)
    RGB = torch.where(H >= 5/6, torch.cat((C, z, X), dim=dim), RGB)
    return RGB + m
