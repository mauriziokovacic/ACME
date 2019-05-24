from utility.row import *

def polysides(T):
    """
    Returns the number of sides of the n-gons in the topology tensor

    Parameters
    ----------
    T : LongTensor
        the topology tensor

    Returns
    -------
    int
        the number of sides
    """

    return row(T)
