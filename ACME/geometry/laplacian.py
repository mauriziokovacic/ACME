import torch
from ACME.topology.laplacian import *



def combinatorial_Laplacian(P,T):
    """
    Computes the combinatorial laplacian matrix for a given mesh.

    Parameters
    ----------
    P : Tensor
        the input points set tensor
    T : LongTensor
        the topology tensor

    Returns
    -------
    Tensor
        the laplacian matrix
    """

    return laplacian(Adjacency(T,P=P,type='std'))



def cotangent_Laplacian(P,T):
    """
    Computes the cotangent weights laplacian matrix for a given triangle mesh.

    Parameters
    ----------
    P : Tensor
        the input points set tensor
    T : LongTensor
        the topology tensor

    Returns
    -------
    Tensor
        the laplacian matrix
    """

    return laplacian(Adjacency(T,P=P,type='cot'))
