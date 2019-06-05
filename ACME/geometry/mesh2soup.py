import torch
from utility.row     import *
from utility.numel   import *
from utility.indices import *

def mesh2soup(P,T):
    """
    Converts the given mesh into a polygon soup

    Parameters
    ----------
    P : Tensor
        the input point set tensor
    T : LongTensor
        the topology tensor

    Returns
    -------
    (Tensor,LongTensor)
        the vertices tensor and the topology tensor
    """

    n = col(T)
    v = torch.reshape(T,numel(T),1)
    return P[v], torch.reshape(indices(0,row(P)-1,device=T.device),n,numel(v)/n)
