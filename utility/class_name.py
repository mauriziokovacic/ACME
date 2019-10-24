def class_name(obj):
    """
    Returns the name of the object class

    Parameters
    ----------
    obj : object
        a object

    Returns
    -------
    str
        the name of the object class
    """

    return obj.__class__.__name__
