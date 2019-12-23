from .cyan_channel    import *
from .magenta_channel import *
from .yellow_channel  import *
from .black_channel   import *


def cmyk2rgb(color, dim=-1):
    """
    Converts the given CMYK colors in RGB format

    Parameters
    ----------
    C : Tensor
        the RGB color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the RGB color tensor
    """

    C = cyan_channel(color, dim=dim)
    M = magenta_channel(color, dim=dim)
    Y = yellow_channel(color, dim=dim)
    K = black_channel(color, dim=dim)
    R = (1 - C) * (1 - K)
    G = (1 - M) * (1 - K)
    B = (1 - Y) * (1 - K)
    return torch.cat((R, G, B), dim=dim)
