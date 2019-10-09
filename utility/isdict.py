def isdict(*obj):
    """
    Returns whether or not the input is a dict

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are dicts, False otherwise
    """

    return all([isinstance(o, dict) for o in obj])
