def islist(*obj):
    """
    Returns whether or not the input is a list

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are lists, False otherwise
    """

    return all([isinstance(o, list) for o in obj])
