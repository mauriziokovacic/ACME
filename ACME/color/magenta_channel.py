from .channel import *


def magenta_channel(C, dim=-1):
    """
    Extracts the magenta channel from colors in CMYK format

    Parameters
    ----------
    C : Tensor
        the CMYK color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the magenta tensor
    """

    return channel(C, 1, dim=dim)
