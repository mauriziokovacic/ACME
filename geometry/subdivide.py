import torch
from ..topology.ispoly      import *
from ..topology.subdivision import *
from .soup2mesh             import *


def subdivide(P, T, iter=1):
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
    (Tensor, LongTensor, Tensor)
        the new points set, the new topology and the subdivision matrix
    """

    if istri(T):
        fun = xtri
    else:
        if isquad(T):
            fun = xquad
        else:
            assert False, 'Topology not supported yet'
    M, t    = fun(T, iter=iter)
    p       = torch.matmul(M, P)
    p, t, I = soup2mesh(p, t)[0:3]
    return p, t, M[I]
