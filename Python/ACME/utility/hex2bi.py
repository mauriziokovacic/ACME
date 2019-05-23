from . import de2bi

def hex2bi(obj):
    """
    Converts a hexadecimal number into its binary representation

    Parameters
    ----------
    obj : str
        the hexadecimal number

    Returns
    -------
    int
        the binary representation of the input number
    """

    return de2bi(int(str(obj),16))
