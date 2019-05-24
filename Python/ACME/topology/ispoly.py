from utility.row import *

def ispoly(T,n=None):
    """
    Returns whether or not the input topology is composed of n-gons.
    If n is None then the function checks if the topology is a composed of
    generic n-gons, rather than edges, triangles or quads.

    Parameters
    ----------
    T : LongTensor
        the topology tensor
    n : int (optional)
        the target number of polygon sides (default is None)

    Returns
    -------
    bool
        True if topology has n-gons, False otherwise
    """

    if n is None:
        return row(T)>4
    return row(T)==n



def isedge(T):
    """
    Returns whether or not the input topology is an edge tensor

    Parameters
    ----------
    T : LongTensor
        the topology tensor

    Returns
    -------
    bool
        True if topology is an edge tensor, False otherwise
    """

    return ispoly(T,2)



def istri(T):
    """
    Returns whether or not the input topology is a triangle tensor

    Parameters
    ----------
    T : LongTensor
        the topology tensor

    Returns
    -------
    bool
        True if topology is a triangle tensor, False otherwise
    """

    return ispoly(T,3)



def isquad(T):
    """
    Returns whether or not the input topology is a quad tensor

    Parameters
    ----------
    T : LongTensor
        the topology tensor

    Returns
    -------
    bool
        True if topology is a quad tensor, False otherwise
    """

    return ispoly(T,4)



def istet(T):
    """
    Returns whether or not the input topology is a tetrahedral tensor

    Parameters
    ----------
    T : LongTensor
        the topology tensor

    Returns
    -------
    bool
        True if topology is a tetrahedral tensor, False otherwise
    """

    return ispoly(T,4)



def ishex(T):
    """
    Returns whether or not the input topology is a hexahedral tensor

    Parameters
    ----------
    T : LongTensor
        the topology tensor

    Returns
    -------
    bool
        True if topology is a hexahedral tensor, False otherwise
    """

    return ispoly(T,6)
