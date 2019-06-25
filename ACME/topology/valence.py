import torch
from ..utility.unique     import *
from ..utility.accumarray import *
from .poly2edge           import *

def valence(T,n=None):
    """
    Returns the valence of the vertices of the input topology

    Parameters
    ----------
    T : LongTensor
        the topology tensor

    Returns
    -------
    LongTensor
        the vertex valence
    """

    if n is None:
        n = torch.max(T)[0]
    E = unique(poly2edge(poly2edge(T)[0])[0],ByRows=True)[0]
    return accumarray(E[0],1);

    #return torch.sum(Adjacency(T,dtype=torch.long),dim=1,keepdim=True)
