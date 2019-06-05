import torch
from utility.row     import *
from utility.col     import *
from utility.strcmpi import *
from math.eye        import *
from .adjacency      import *
from .degree         import *



def Laplacian(A,type='std'):
    """
    Returns the laplacian (standard,symmetric or random walk) matrix from a given adjacency matrix

    Parameters
    ----------
    A : Tensor
        an adjacency matrix
    type : str (optional)
        the laplacian matrix type. It can either be 'std','sym' or 'walk' (default is 'std')

    Returns
    -------
    Tensor
        the laplacian matrix

    Raises
    ------
    AssertionError
        if type is not supported
    """

    D = Degree(A)
    if strcmpi(type,'std'):
        return D-A
    I  = eye(row(D),col(D),device=A.device)
    iD = torch.diag(torch.reciprocal(torch.diag(D)))
    if strcmpi(type,'sym'):
        iD = torch.sqrt(iD)
        return I-torch.mm(iD,torch.mm(A,iD))
    if strmpi(type,'walk'):
        return torch.mm(iD,A)
    assert False, 'Unknown type'



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

    return Laplacian(Adjacency(T,P=P,type='std'),type='std')



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

    return Laplacian(Adjacency(T,P=P,type='cot'),type='std')
