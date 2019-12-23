from .red_channel import *
from .green_channel import *
from .blue_channel import *


def rgb2cmyk(color, dim=-1):
    """
    Converts the given RGB colors in CMYK format

    Parameters
    ----------
    color : Tensor
        the RGB color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the CMYK color tensor
    """

    R = red_channel(color, dim=dim)
    G = green_channel(color, dim=dim)
    B = blue_channel(color, dim=dim)
    K = 1 - torch.max(torch.max(R, G), B)
    C = (1 - R - K) / (1 - K)
    M = (1 - G - K) / (1 - K)
    Y = (1 - B - K) / (1 - K)
    return torch.cat((C, M, Y, K), dim=dim)
