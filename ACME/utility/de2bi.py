def de2bi(obj):
    """
    Converts a decimal number into its binary representation

    Parameters
    ----------
    obj : int or str
        A number in decimal format

    Returns
    -------
    int
        The binary representation of the input number
    """

    return int(bin(int(obj))[2:])
