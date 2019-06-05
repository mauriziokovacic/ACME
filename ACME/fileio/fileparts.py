import os

def fileparts(filename):
    """
    Splits the input file name string into path,name and extension

    Parameters
    ----------
    filename : str
        a file name

    Returns
    -------
    (str,str,str)
        the path, name and extension of the input file
    """

    path, name = os.path.split(filename)
    name, ext  = os.path.splitext(name)
    return path,name,ext
