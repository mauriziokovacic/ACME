def istype(type, *obj):
    """
    Returns whether or not all the inputs are of the specified type

    Parameters
    ----------
    type : type
        the type to check against to
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are of the specified type, False otherwise
    """

    return all([isinstance(o, type) for o in obj])
