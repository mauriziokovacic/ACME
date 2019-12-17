from .color2float   import *
from .red_channel   import *
from .green_channel import *
from .blue_channel  import *


def rgb2hsv(C, dim=-1):
    """
    Converts the given RGB colors in HSV format

    Parameters
    ----------
    C : Tensor
        the RGB color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the HSV color tensor
    """

    c     = color2float(C)
    cmin  = torch.min(c, dim=-1, keepdim=True)[0]
    cmax  = torch.max(c, dim=-1, keepdim=True)[0]
    delta = cmax - cmin
    R = red_channel(c, dim=dim)
    G = green_channel(c, dim=dim)
    B = blue_channel(c, dim=dim)
    H = torch.zeros_like(cmax)
    H = torch.where(cmax == R, (60 * ((G - B) / delta) % 6) / 360, H)
    H = torch.where(cmax == G, (60 * ((B - R) / delta) + 2) / 360, H)
    H = torch.where(cmax == B, (60 * ((R - G) / delta) + 4) / 360, H)
    S = torch.where(cmax == 0, torch.zeros_like(cmax), delta/cmax)
    V = cmax
    return torch.cat((H, S, V), dim=dim)
