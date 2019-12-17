from .channel import *


def green_channel(C, dim=-1):
    """
    Extracts the green channel from colors in RGB format

    Parameters
    ----------
    C : Tensor
        the RGB color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the green tensor
    """

    return channel(C, 1, dim=dim)
