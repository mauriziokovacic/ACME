from .channel import *


def yellow_channel(C, dim=-1):
    """
    Extracts the yellow channel from colors in CMYK format

    Parameters
    ----------
    C : Tensor
        the CMYK color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the yellow tensor
    """

    return channel(C, 2, dim=dim)
