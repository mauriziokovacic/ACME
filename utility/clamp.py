import math
import torch
from .istorch import *
from .size import *
from .ndim import *


def clamp(a, inf=0, sup=1):
    """
    Returns the clamped value of the input in range [inf,sup]

    Parameters
    ----------
    a : int,float or Tensor
        input value
    inf : int or float (optional)
        the inferior bound (default is 0)
    sup : int or float (optional)
        the superior bound (default is 1)

    Returns
    -------
    int or float or Tensor
        the clamped input
    """

    if istorch(a):
        return torch.clamp(a, inf, sup)
    return min(max(inf, a), sup)


def clamp_max(a, sup=1):
    """
    Returns the clamped value of the input in range [-\infty,sup]

    Parameters
    ----------
    a : int,float or Tensor
        input value
    sup : int or float (optional)
        the superior bound (default is 1)

    Returns
    -------
    int or float or Tensor
        the clamped input
    """

    return clamp(a, inf=-math.inf, sup=sup)


def clamp_min(a, inf=0):
    """
    Returns the clamped value of the input in range [inf,\infty]

    Parameters
    ----------
    a : int,float or Tensor
        input value
    inf : int or float (optional)
        the inferior bound (default is 0)

    Returns
    -------
    int or float or Tensor
        the clamped input
    """

    return clamp(a, inf=inf, sup=math.inf)

