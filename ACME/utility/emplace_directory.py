import os


def emplace_directory(path, name):
    """
    Creates the directory specified by name in the given path if the directory does not exists

    Parameters
    ----------
    path : str
        the path where to create the directory
    name : str
        the directory name

    Returns
    -------
    None
    """

    dir = os.path.join(path, name)
    if not os.path.exists(dir):
        os.makedirs(dir)
