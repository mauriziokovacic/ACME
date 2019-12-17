from .channel import *


def depth_channel(C, dim=-1):
    """
    Extracts the depth channel from colors in RGBD format

    Parameters
    ----------
    C : Tensor
        the RGBD color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the depth tensor
    """

    return channel(C, 3, dim=dim)
