def isint(*obj):
    """
    Returns whether or not the input is an int

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are int, False otherwise
    """

    return all([isinstance(o, int) for o in obj])
