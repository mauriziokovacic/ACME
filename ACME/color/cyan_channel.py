from .channel import *


def cyan_channel(C, dim=-1):
    """
    Extracts the cyan channel from colors in CMYK format

    Parameters
    ----------
    C : Tensor
        the CMYK color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the cyan tensor
    """

    return channel(C, 0, dim=dim)
