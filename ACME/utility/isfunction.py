def isfunction(*obj):
    """
    Returns whether or not the input is a function

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are functions, False otherwise
    """

    return all([callable(o) for o in obj])
