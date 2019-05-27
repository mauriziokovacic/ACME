import torch
from .adjacency import *

def valence(T):
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

    return torch.sum(Adjacency(T,dtype=torch.long),dim=1,keepdim=True)
