def hex2de(obj):
    """
    Converts a hexadeciaml number into its decimal representation

    Parameters
    ----------
    obj : int or str
        A number in hexadeciaml format

    Returns
    -------
    int
        the decimal representation of the input number
    """

    return int(str(obj), 16)
