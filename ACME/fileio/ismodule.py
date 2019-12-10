import os


def ismodule(path):
    """
    Returns True if the given path is a Python module, False otherwise

    Parameters
    ----------
    path : str
        the path to check

    Returns
    -------
    bool
        True if the given path is a Python module, False otherwise
    """
    
    return os.path.isfile(path) and path.endswith('.py')
