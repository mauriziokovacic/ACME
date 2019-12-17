from .channel import *


def red_channel(C, dim=-1):
    """
    Extracts the red channel from colors in RGB format

    Parameters
    ----------
    C : Tensor
        the RGB color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the red tensor
    """

    return channel(C, 0, dim=dim)
