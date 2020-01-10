from ..utility.Uint8Tensor import *
from .color2float          import *


def black():
    """
    Returns the black RGB tensor

    Returns
    -------
    Tensor
        the (1,3,) black tensor
    """

    return torch.zeros(1, 3)


def white():
    """
    Returns the white RGB tensor

    Returns
    -------
    Tensor
        the (1,3,) white tensor
    """

    return torch.ones(1, 3)


def red():
    """
    Returns the red RGB tensor

    Returns
    -------
    Tensor
        the (1,3,) red tensor
    """

    return color2float(Uint8Tensor([237, 28, 36]))


def green():
    """
    Returns the green RGB tensor

    Returns
    -------
    Tensor
        the (1,3,) green tensor
    """

    return color2float(Uint8Tensor([34, 177, 76]))


def blue():
    """
    Returns the blue RGB tensor

    Returns
    -------
    Tensor
        the (1,3,) blue tensor
    """

    return color2float(Uint8Tensor([0, 162, 232]))


def cyan():
    """
    Returns the cyan RGB tensor

    Returns
    -------
    Tensor
        the (1,3,) cyan tensor
    """

    return torch.add(torch.neg(red()), 1)


def magenta():
    """
    Returns the magenta RGB tensor

    Returns
    -------
    Tensor
        the (1,3,) magenta tensor
    """

    return torch.add(torch.neg(green()), 1)


def yellow():
    """
    Returns the yellow RGB tensor

    Returns
    -------
    Tensor
        the (1,3,) yellow tensor
    """

    return color2float(Uint8Tensor([[255, 242, 0]]))


def brown():
    """
    Returns the brown RGB tensor

    Returns
    -------
    Tensor
        the (1,3,) brown tensor
    """

    return color2float(Uint8Tensor([[149, 116, 83]]))


def dark_teal():
    """
    Returns the dark teal RGB tensor

    Returns
    -------
    Tensor
        the (1,3,) dark teal tensor
    """

    return color2float(Uint8Tensor([[98, 140, 178]]))


def gray():
    """
    Returns the gray RGB tensor

    Returns
    -------
    Tensor
        the (1,3,) gray tensor
    """

    return torch.mul(torch.ones(1, 3), 0.5)


def orange():
    """
    Returns the orange RGB tensor

    Returns
    -------
    Tensor
        the (1,3,) orange tensor
    """

    return color2float(Uint8Tensor([[253, 135, 86]]))


def pink():
    """
    Returns the pink RGB tensor

    Returns
    -------
    Tensor
        the (1,3,) pink tensor
    """

    return color2float(Uint8Tensor([[254, 194, 194]]))


def teal():
    """
    Returns the teal RGB tensor

    Returns
    -------
    Tensor
        the (1,3,) teal tensor
    """

    return color2float(Uint8Tensor([[144, 216, 196]]))
