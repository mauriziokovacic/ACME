from functools import reduce

def prod(*obj):
    """
    Returns the product of the input values

    Parameters
    ----------
    *obj : list of int or float
        a sequence of numbers
    Returns
    -------
    int or float
        the product of the input values
    """

    return reduce((lambda a, b : a*b), obj)
