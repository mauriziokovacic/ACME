import torch
from ACME.utility.row       import *
from ACME.utility.col       import *
from ACME.utility.repmat    import *
from ACME.utility.circshift import *
from ACME.utility.indices   import *
from ACME.utility.find      import *
from .ind2poly              import *

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
    E = torch.cat((torch.cat(tuple(T),                   ).unsqueeze(0),
                   torch.cat(tuple(circshift(T,-1,dim=0))).unsqueeze(0)),dim=0)
    F = repmat(indices(0,col(T)-1,device=T.device),row(T),1)
    return E,F



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



def adj2edge(A):
    """
    Extracts the edges from an adjacency matrix.

    Parameters
    ----------
    A : Tensor
        the adjacency matrix

    Returns
    -------
    LongTensor
        the edge tensor
    """

    i,j = find(A>0,linear=False)
    return ind2edge(i,j)
