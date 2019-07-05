def isstring(*obj):
    """
    Returns whether or not the input is a string

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are strings, False otherwise
    """

    return all([isinstance(o, str) for o in obj])
