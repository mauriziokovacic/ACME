import torch
from ..topology.ispoly      import *
from ..topology.subdivision import *
from .soup2mesh             import *

def subdivide(P,T,iter=1):
    """
    Subdivides the given mesh n times

    Parameters
    ----------
    P : Tensor
        the input points set
    T : LongTensor
        the topology tensor
    iter : int (optional)
        the number of times to subdivide the input mesh (default is 1)

    Returns
    -------
    (Tensor,LongTensor)
        the new points set and the new topology
    """

    if istri(T):
        fun = xtri
    else:
        if isquad(T):
            fun = xquad
        else:
            assert False,'Topology not supported yet'
    M,T = fun(T,iter=iter)
    P   = torch.mm(M,P)
    P,T = soup2mesh(P,T)[0:2]
    return P,T
