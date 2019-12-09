from .bi2de import *
from .de2hex import *


def bi2hex(obj):
    """
    Converts a binary number into its hexdecimal representation

    Parameters
    ----------
    obj : int or str
        A number in binary format

    Returns
    -------
    str
        the hexdecimal representation of the input number
    """

    return de2hex(bi2de(obj))
