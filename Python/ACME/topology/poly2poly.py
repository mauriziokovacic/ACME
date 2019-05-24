import torch
from utility.row    import *
from utility.repmat import *

def poly2poly(T,n):
    """
    Reshapes the input topology tensor into n-gons

    Parameters
    ----------
    T : LongTensor
        the topology tensor
    n : int
        target number of polygon sides

    Returns
    -------
    LongTensor
        the new topology tensor
    """

    t = repmat(T,n//row(T)+1,1)
    t = t[:-(row(T)-n%row(T))]
    return t



def edge2poly(E,n):
    """
    Reshapes the input edge tensor into n-gons

    Parameters
    ----------
    T : LongTensor
        the edge tensor
    n : int
        target number of polygon sides

    Returns
    -------
    LongTensor
        the new topology tensor
    """

    return poly2poly(E,n)



def tri2poly(T,n):
    """
    Reshapes the input triangle tensor into n-gons

    Parameters
    ----------
    T : LongTensor
        the triangle tensor
    n : int
        target number of polygon sides

    Returns
    -------
    LongTensor
        the new topology tensor
    """

    return poly2poly(T,n)



def quad2poly(T,n):
    """
    Reshapes the input quad tensor into n-gons

    Parameters
    ----------
    T : LongTensor
        the quad tensor
    n : int
        target number of polygon sides

    Returns
    -------
    LongTensor
        the new topology tensor
    """

    return poly2poly(T,n)



def edge2tri(E):
    """
    Reshapes the input edge tensor into triangles

    Parameters
    ----------
    E : LongTensor
        the edge tensor

    Returns
    -------
    LongTensor
        the triangle tensor
    """

    return edge2poly(E,3)



def edge2quad(E):
    """
    Reshapes the input edge tensor into quads

    Parameters
    ----------
    E : LongTensor
        the edge tensor

    Returns
    -------
    LongTensor
        the quad tensor
    """

    return edge2poly(E,4)



def quad2tri(T):
    """
    Subdivides the input quad tensor into triangles

    Parameters
    ----------
    T : LongTensor
        the quad tensor

    Returns
    -------
    LongTensor
        the new triangle tensor
    """

    I = poly2ind(T)
    return torch.cat((torch.cat((I[0],I[1],I[2]),dim=0).unsqueeze(1),torch.cat((I[2],I[3],I[0]),dim=0).unsqueeze(1)),dim=1)

