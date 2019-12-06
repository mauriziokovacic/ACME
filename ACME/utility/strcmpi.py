from .strcmp import *

def strcmpi(str_a, str_b):
    """
    Case insensitive comparison of two strings

    Parameters
    ----------
    str_a : str
        first string
    str_b : str
        second string

    Returns
    -------
    bool
        True if the two strings are identical, False otherwise
    """

    return strcmp(str_a.lower(), str_b.lower())
