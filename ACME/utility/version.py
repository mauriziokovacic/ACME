import sys


def version():
    """
    Returns the Python version currently used

    Returns
    -------
    str
        the Python version currently used
    """
    return '{}.{}.{}'.format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
