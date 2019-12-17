from .channel import *


def blue_channel(C, dim=-1):
    """
    Extracts the blue channel from colors in RGB format

    Parameters
    ----------
    C : Tensor
        the RGB color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the blue tensor
    """

    return channel(C, 2, dim=dim)
