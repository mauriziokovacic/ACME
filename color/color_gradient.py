from ..utility.linspace import *
from .fetch_texture     import *


def color_gradient(cdata, cres=64):
    """
    Returns a gradient consisting of cres colors, from the input color data

    Parameters
    ----------
    cdata : Tensor
        a (N,3,) color tensor
    cres : int (optional)
        the number of output colors

    Returns
    -------
    Tensor
        a (cres,3,) color tensor
    """

    return fetch_texture1D(cdata, linspace(0, 1, cres))
