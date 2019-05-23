from . import bi2de
from . import de2hex

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
