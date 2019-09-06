from .constant import *


def deg2rad(value):
    """
    Converts degree values into radians

    Parameters
    ----------
    value : int,float or Tensor
        the degree values

    Returns
    -------
    float or Tensor
        the radians values
    """

    return (value*PI)/180
