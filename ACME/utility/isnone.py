def isnone(*obj):
    """
    Returns whether or not the input is None

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are None, False otherwise
    """

    return all([o is None for o in obj])
