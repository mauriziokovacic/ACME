from .constant import *


def rad2deg(value):
    """
    Computes the degree value from input radians

    Parameters
    ----------
    value : int,float or Tensor
        input radians values

    Returns
    -------
    float or Tensor
        the degree values
    """

    return (value*180)/PI
