import os


def is_package(path):
    """
    Returns True if the given path is a Python package, False otherwise

    Parameters
    ----------
    path : str
        the path to check

    Returns
    -------
    bool
        True if the given path is a Python package, False otherwise
    """

    return os.path.isdir(path) and ('__init__.py' in os.listdir(path))
