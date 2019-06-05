from .isint     import *
from .isfloat   import *
from .iscomplex import *

def isscalar(*obj):
    """
    Returns whether or not the input is a scalar

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are scalars, False otherwise
    """

    return isint(*obj) or isfloat(*obj) or iscomplex(*obj)
