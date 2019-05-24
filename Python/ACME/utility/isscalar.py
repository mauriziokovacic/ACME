from .isint     import *
from .isfloat   import *
from .iscomplex import *

def isscalar(obj):
    """
    Returns whether or not the input is a scalar

    Parameters
    ----------
    obj : object
        any object

    Returns
    -------
    bool
        True if the input is scalar, False otherwise
    """

    return isint(obj) or isfloat(obj) or iscomplex(obj)
