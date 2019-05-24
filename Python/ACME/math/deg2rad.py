from .constant import *

def deg2rad(degree):
    """
    Converts degree values into radians

    Parameters
    ----------
    degree : int,float or Tensor
        the degree values

    Returns
    -------
    float or Tensor
        the radians values
    """

    return (value*PI)/180
