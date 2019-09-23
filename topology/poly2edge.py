import torch
from ..utility.row       import *
from ..utility.col       import *
from ..utility.repmat    import *
from ..utility.circshift import *
from ..utility.indices   import *
from .poly2unique        import *


def poly2edge(T):
    """
    Extracts the edges from the polygon tensor.

    Parameters
    ----------
    T : LongTensor
        the topology tensor

    Returns
    -------
    (LongTensor,LongTensor)
        the edge tensor and the respective polygon indices

    Raises
    ------
    AssertionError
        if the topology tensor is not at least (2,n)
    """

    assert row(T)>1, "Topology matrix should be at least of size 2xt"
    E = torch.cat((torch.cat(tuple(T),                     ).unsqueeze(0),
                   torch.cat(tuple(circshift(T, -1, dim=0))).unsqueeze(0)), dim=0)
    F = repmat(indices(0, col(T)-1, device=T.device), row(T), 1)
    return E, F


def tri2edge(T):
    """
    Extracts the edges from the triangle tensor.

    Parameters
    ----------
    T : LongTensor
        the topology tensor

    Returns
    -------
    (LongTensor,LongTensor)
        the edge tensor and the respective triangle indices
    """

    return poly2edge(T)


def quad2edge(T):
    """
    Extracts the edges from the quad tensor.

    Parameters
    ----------
    T : LongTensor
        the topology tensor

    Returns
    -------
    (LongTensor,LongTensor)
        the edge tensor and the respective quad indices
    """
    return poly2edge(T)


def edge2undirect(E):
    return poly2edge(poly2unique(E))[0]


def poly2undirect(T):
    return edge2undirect(poly2edge(T)[0])
