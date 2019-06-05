from .floor import *

def sawtooth(x):
    """
    Returns the sawtooth function value for input x

    Parameters
    ----------
    x : scalar or Tensor
        the input value

    Returns
    -------
    scalar or Tensor
        the sawtooth function value
    """

    return 1-x+floor(x)
