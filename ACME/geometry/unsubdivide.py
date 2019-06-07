import torch
from utility.row     import *
from utility.col     import *
from utility.reindex import *
from utility.unique  import *
from topology.ispoly import *


def unsubdivide(P,T,iter=1):
    """
    Unsubdivides the given mesh n times

    In order to work, the mesh is intended subdivided using the method 'subdivide'.

    Parameters
    ----------
    P : Tensor
        the input points set
    T : LongTensor
        the topology tensor
    iter : int (optional)
        the number of times to unsubdivide the input mesh (default is 1)

    Returns
    -------
    (Tensor,LongTensor)
        the new points set and the new topology
    """

    for n in range(0,iter):
        i = torch.t(torch.reshape(T[0],col(T)//4,4))
        if(istri(T)):
            i = i[0:3]
        p = P[unique(i)]
        t = reindex(i)
        if (row(t)==1):
            t = torch.t(t)
    return p,t
