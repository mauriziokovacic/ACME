def isbool(*obj):
    """
    Returns whether or not the input is a bool

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are bool, False otherwise
    """

    return all([isinstance(o,bool) for o in obj])
