from ..utility.LongTensor import *
from ..utility.repmat     import *
from ..utility.circshift  import *
from ..utility.indices    import *
from .poly2unique         import *


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

    assert row(T) > 1, "Topology matrix should be at least of size 2xt"
    E = torch.cat((T.contiguous().view(-1).unsqueeze(0),
                   torch.cat(tuple(circshift(T, -1, dim=0))).unsqueeze(0)), dim=0)
    F = repmat(indices(0, T.shape[1]-1, device=T.device), T.shape[0], 1)
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
    return poly2edge(poly2unique(E)[0])[0]


def poly2undirect(T):
    return edge2undirect(poly2edge(T)[0])


def tet2edge(T):
    e = LongTensor([[0, 0, 0, 1, 1, 2],
                    [1, 2, 3, 2, 3, 3]]).t()
    return torch.sort(T[e].permute(1, 0, 2).contiguous().view(2, -1), dim=0)[0]


def hex2edge(T):
    e = LongTensor([[0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6],
                    [1, 3, 4, 2, 5, 3, 6, 7, 5, 7, 6, 7]]).t()
    return torch.sort(T[e].permute(1, 0, 2).contiguous().view(2, -1), dim=0)[0]
