def de2hex(obj):
    """
    Converts a decimal number into its hexadecimal representation

    Parameters
    ----------
    obj : int or str
        A number in decimal format

    Returns
    -------
    str
        The hexadecimal representation of the input number
    """

    return hex(int(obj))[2:].upper()
