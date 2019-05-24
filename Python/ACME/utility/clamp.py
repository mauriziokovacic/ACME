from .size import *
from .ndim import *

def clamp(a,inf=0,sup=1):
    """
    Returns the clamped value of the input in range [inf-sup]

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
        return torch.clamp(a,inf,sup)
    return min(max(inf,a),sup)
