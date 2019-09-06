def isfloat(*obj):
    """
    Returns whether or not the input is a float

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are float, False otherwise
    """

    return all([isinstance(o, float) for o in obj])
