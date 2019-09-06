from .istensor import *
from .flatten  import *

def isnan(*obj):
    """
    Returns whether or not the input is nan

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are nan, False otherwise
    """

    out = [flatten(o) if istensor(o) else o for o in obj]
    return any([any(o != o) if istensor(o) else o != o for o in out])
