from .inf      import *
from .istensor import *
from .flatten  import *


def isinf(*obj):
    """
    Returns whether or not the input is nan

    Parameters
    ----------
    a : int or float or tensor
        an input

    Returns
    -------
    bool
        True if the input is infinite, False otherwise
    """

    out = [flatten(o) if istensor(o) else o for o in obj]
    return any([any(o == Inf) if istensor(o) else o == Inf for o in out])
