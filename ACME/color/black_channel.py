from .channel import *


def black_channel(C, dim=-1):
    """
    Extracts the black channel from colors in CMYK format

    Parameters
    ----------
    C : Tensor
        the CMYK color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the black tensor
    """

    return channel(C, 3, dim=dim)
