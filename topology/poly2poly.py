from ..utility.LongTensor import *
from ..utility.indices    import *
from ..utility.repmat     import *
from ..utility.unique     import *
from .poly2ind            import *


def poly2poly(T, n):
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

    t = repmat(T, n // row(T) + 1, 1)
    t = t[:-(row(T) - n % row(T))]
    return t


def edge2poly(E, n):
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

    return poly2poly(E, n)


def tri2poly(T, n):
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

    return poly2poly(T, n)


def quad2poly(T, n):
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

    return poly2poly(T, n)


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

    return edge2poly(E, 3)


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

    return edge2poly(E, 4)


def poly2tri(P):
    """
    Converts a given polygon tensor into triangles, using a naive approach

    A polygon will be split by connecting two consecutive nodes with the first one

    Parameters
    ----------
    P : LongTensor
        the topology tensor

    Returns
    -------
    (LongTensor,LongTensor)
        the triangle tensor and the index of the polygon generating it
    """

    i = torch.zeros(row(P) - 2, 1, dtype=P.dtype, device=P.device)
    j = indices(1, row(P) - 2, device=P.device)
    k = (j + 1) % row(P)
    t = torch.cat((i, j, k), dim=1)
    T = tuple(P[t])
    I = repmat(indices(0, col(P) - 1, device=P.device), (len(T), 1))
    T = torch.cat(T, dim=1)
    return T, I


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

    i = LongTensor([[0, 1, 2], [0, 2, 3]], device=T.device).t()
    return T[i].contiguous().view(3, -1)


def poly2node(T):
    """
    Returns the unique nodes of the input topology

    Parameters
    ----------
    T : LongTensor
        the topology tensor

    Returns
    -------
    LongTensor
        the unique nodes indices
    """

    return unique(T.flatten())[0]


def hex2tet(T):
    """
    Converts the input hexaedra into a tetrahedra topology

    Parameters
    ----------
    T : LongTensor
        the (8,N,) topology tensor

    Returns
    -------
    LongTensor
        the new (4,M,) topology tensor
    """

    i = LongTensor([[0, 1, 3, 4],
                    [2, 6, 3, 1],
                    [3, 6, 7, 4],
                    [1, 6, 3, 4],
                    [5, 4, 6, 1]], device=T.device).t()
    return T[i].contiguous().view(4, -1)


def tet2tri(T):
    """
    Converts the input tetrahedra into triangular faces

    Parameters
    ----------
    T : LongTensor
        the (4,N,) topology tensor

    Returns
    -------
    LongTensor
        the new (3,M,) topology tensor
    """

    i = LongTensor([[0, 2, 1],
                    [1, 2, 3],
                    [3, 2, 0],
                    [0, 1, 3]], device=T.device).t()
    return T[i].contiguous().view(3, -1)


def hex2quad(T):
    """
    Converts the input hexaedra into quad faces

    Parameters
    ----------
    T : LongTensor
        the (8,N,) topology tensor

    Returns
    -------
    LongTensor
        the new (4,M,) topology tensor
    """

    i = LongTensor([[0, 1, 2, 3],
                    [4, 0, 3, 7],
                    [5, 4, 7, 6],
                    [1, 5, 6, 2],
                    [3, 2, 6, 7],
                    [4, 5, 1, 0]], device=T.device).t()
    return T[i].contiguous().view(4, -1)
